# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss
from transformers import Trainer
import math

# ================================================================================ #
# [MSKIM] new Trainer for QAT-KD
# ================================================================================ #

class KDTrainer(Trainer):
    def __init__(self, teacher_model, quant_args, *args, **kwargs):
        super(KDTrainer, self).__init__(*args, **kwargs)
        
        self.quant_args = quant_args

        if quant_args.kd_qat_full:
            self.teacher_model = teacher_model
            self.teacher_model = self.teacher_model.eval()
        else:
            self.teacher_model = None

    # Implement KD functions
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # Training / Evaluation 
        is_eval = True if not model.training else False

        ce_loss = CrossEntropyLoss()
        ce_loss_none = CrossEntropyLoss(reduction="none")
        mse_loss = MSELoss()
        
        metrics = {}
        gt_loss = 0; pred_loss = 0; l2l_loss = 0; tsld_loss = 0
    
        if self.quant_args.kd_l2l: # [WARNING] This option incurs painful memory usage in training! (especially in LLM)
            inputs["output_attentions"] = True; inputs["output_hidden_states"] = True

        # Teacher Model Inference (KD)
        if self.quant_args.kd_qat_full and not is_eval:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)

            teacher_loss = teacher_outputs["loss"] if isinstance(teacher_outputs, dict) else teacher_outputs[0]
            teacher_logits = teacher_outputs["logits"].float() if isinstance(teacher_outputs, dict) else teacher_outputs[1]

        # Student Model Inference
        outputs = self.model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        # training gt loss
        if self.quant_args.kd_gt and not is_eval:
            gt_loss = loss
            metrics["gt_loss"] = gt_loss.item()

        # logit distillation
        if self.quant_args.kd_pred and not is_eval:
            shift_logits = logits[..., :-1, :].contiguous() 
            tc_shift_logits = teacher_logits[..., :-1, :].contiguous() 

            student_likelihood = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            targets_prob = torch.nn.functional.softmax(tc_shift_logits, dim=-1)
            pred_loss =  torch.sum((- targets_prob * student_likelihood), dim=-1).mean()
            metrics["pred_loss"] = pred_loss.item()

        # token-scaled logit distillation (Kim et al, https://arxiv.org/abs/2308.06744)
        if self.quant_args.kd_tsld and not is_eval:
            shift_logits = logits[..., :-1, :].contiguous() 
            tc_shift_logits = teacher_logits[..., :-1, :].contiguous() 
            
            # Step 1. get per-token ce loss with teacher logits
            tc_shift_labels = inputs["labels"][..., 1:].contiguous().to(inputs["labels"].device)
            tc_loss_all = ce_loss_none(tc_shift_logits.view(-1,tc_shift_logits.size(-1)), tc_shift_labels.view(-1))

            # Step 2. get token-scale with tc_loss_all and temperatured softmax function
            tc_all = tc_loss_all.reshape(tc_shift_logits.shape[0], -1)
            token_scale = torch.nn.functional.softmax(tc_all / self.quant_args.kd_tsld_temp, dim=-1).clone().detach()

            # Step 3. logit distillation with token-scale
            student_likelihood = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            targets_prob = torch.nn.functional.softmax(tc_shift_logits, dim=-1)
            tsld_loss = (torch.sum((- targets_prob * student_likelihood), dim=-1) * token_scale).sum() # SUM
            # tsld_loss = (torch.sum((- targets_prob * student_likelihood), dim=-1) * token_scale).sum(dim=-1).mean()
            metrics["tsld_loss"] = tsld_loss.item()
        
        # Layer-to-Layer Distillation (Zhang et al, https://aclanthology.org/2020.emnlp-main.37/)
        if self.quant_args.kd_l2l and not is_eval:
            rep_loss = 0
            attnscore_loss = 0

            # Per-layer Transformer Output
            student_reps = outputs.hidden_states
            teacher_reps = teacher_outputs.hidden_states

            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                tmp_loss = mse_loss(student_rep, teacher_rep.float())
                rep_loss += tmp_loss
            
            # Per-layer Attention Score (Query x Value)
            student_scores = outputs.attentions
            teacher_scores = teacher_outputs.attentions
            
            for st_score, tc_score in zip(student_scores, teacher_scores):
                            
                st_score = torch.where(st_score <= -1e5, 0, st_score)
                tc_score = torch.where(tc_score <= -1e5, 0, tc_score)

                # Consider causal attention (upper-traiangular part masked)
                mask = torch.tril(torch.ones_like(tc_score.float())).bool()
                diff = tc_score - st_score
                masked_diff = diff[mask]
                tmp_loss = (masked_diff ** 2).mean()
                attnscore_loss += tmp_loss

            l2l_loss = rep_loss + attnscore_loss 
            metrics["rep_loss"] = rep_loss.item()
            metrics["attn_loss"] = attnscore_loss.item()

        if not is_eval:
            loss = gt_loss + pred_loss + tsld_loss + l2l_loss if self.is_in_train else loss
        
        metrics["total_loss"] = loss.item()

        if self.state.global_step % self.args.logging_steps == 0:
            for n, p in self.model.named_parameters():
                if "scale" in n:
                    print(f"{n[15:]} - {p.max().item():.3f} {p.min().item():.3f} {p.abs().mean().item():.3f}")
            self.log(metrics)
        if not math.isfinite(loss.item()):
            print("Loss is NAN, stopping training")
            import pdb; pdb.set_trace()
        
        return (loss, outputs) if return_outputs else loss


# ================================================================================ #
# Class for Quantizer
# ================================================================================ #

class MatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, A, B):
        device = self.dummy_param.device
        return A @ B

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, max_scale):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, max_scale):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta


        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

class AsymWeightQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, w, quant_args):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(w)
        org_w_shape = w.shape
        q_group_size = quant_args.group_size
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        else:
            w = w.reshape(-1, w.shape[-1])

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** quant_args.n_bits_w - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        
        w = (torch.clamp(torch.round(w / scales) +
                        zeros, min_int, max_int) - zeros) * scales
        
        assert torch.isnan(w).sum() == 0

        w_q = w.reshape(org_w_shape)
            
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None

class AsymActQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, w, quant_args):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(w)
        org_w_shape = w.shape
        q_group_size = quant_args.group_size

        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        else:
            w = w.reshape(-1, w.shape[-1])

        max_val = w.amax(dim=0, keepdim=True) # Token-wise Quantization
        min_val = w.amin(dim=0, keepdim=True)

        max_int = 2 ** quant_args.n_bits_w - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        
        w = (torch.clamp(torch.round(w / scales) +
                        zeros, min_int, max_int) - zeros) * scales
        
        assert torch.isnan(w).sum() == 0

        w_q = w.reshape(org_w_shape)
            
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class PactQuantizer(torch.nn.Module):
    """QuantGPT (ACL 2022)
    Ref: https://arxiv.org/pdf/2203.10705.pdf
    """

    # @staticmethod
    def forward(self, input, quant_args, scale):
        """
        :param input: tensor to be quantized (2-bit Advanced PACT)
        :return: quantized tensor
        """
        
        scale_gamma = scale
        input_ = input

        if quant_args.per_tensor:
            m = input.norm(p=1).div(input.nelement())
            # m = input.max()
            clip_alpha = m * scale_gamma
        else:
            m = input.norm(p=1,dim=1).div(input[0].nelement())
            m = m.expand(input.shape[1], -1).transpose(0,1)
            clip_alpha = m * scale_gamma
            # clip_alpha = clip_alpha.expand(input.shape[1], -1).transpose(0,1)
        
        # clip operation
        try:
            input = torch.where(input <= clip_alpha, input, clip_alpha)
            input = torch.where(input >= -1*clip_alpha, input, -1*clip_alpha)
        except:
            import pdb; pdb.set_trace()
        
        # quantization
        n = 2 ** (quant_args.n_bits_w - 1) - 1
        u = input / clip_alpha
        q_u = round_ste(u * n) / n
        result = q_u * clip_alpha
        # step_size = (clip_alpha * 2) / (2 ** num_bits - 1)
        # result = torch.round(input / step_size) * step_size
        return result


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, quant_args, max_scale=0.7):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        
        org_w_shape = input.shape
        q_group_size = quant_args.group_size

        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            input = input.reshape(-1, q_group_size)
        else:
            input = input.reshape(-1, input.shape[-1])
        
        if quant_args.per_tensor: assert q_group_size == -1, "Conflict with Per Tensor and Per Group Quant!"

        if quant_args.per_tensor:
            # Per Tensor Quantizaiton
            m = input.abs().mean()
            thres = max_scale * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:
            # Per Channel/Group Quantization
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (max_scale * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        result = result.reshape(org_w_shape) # for per-group quantization

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        # input, clip_val = ctx.saved_tensors  # unclipped input
        input = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        # grad_input[input.ge(clip_val[1])] = 0
        # grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs,bias=True, args=None, is_fc2=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.quant_args=args
        self.weight_bits = args.n_bits_w

        if self.quant_args.n_bits_w == 4:
            self.weight_quantizer = AsymWeightQuantizer
        if self.quant_args.n_bits_w == 2:
            if self.quant_args.learned_scale:
                if not self.quant_args.per_tensor:
                    dim = self.weight.shape[0]
                    self.scale = nn.Parameter(torch.ones((dim,1))*self.quant_args.init_scale)
                else:
                    self.scale = nn.Parameter(torch.ones(1)*self.quant_args.init_scale)
                self.weight_quantizer = PactQuantizer()
            else:
                self.weight_quantizer = TwnQuantizer
        
    def forward(self, input):
        
        if self.quant_args.n_bits_w == 4:
            weight = self.weight_quantizer.apply(self.weight, self.quant_args)
        if self.quant_args.n_bits_w == 2:
            if self.quant_args.learned_scale:
                weight = self.weight_quantizer(self.weight, self.quant_args, self.scale)
            else:
                weight = self.weight_quantizer.apply(self.weight, self.quant_args)

        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

class QConv1D(nn.Module):
    def __init__(self, nf, nx, args):
        super().__init__()
        self.quant_args = args        
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        
        if self.quant_args.n_bits_w == 4:
            self.weight_quantizer = AsymWeightQuantizer
        if self.quant_args.n_bits_w == 2:
            self.weight_quantizer = TwnQuantizer

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # Weight Quantization
        weight = self.weight_quantizer.apply(self.weight, self.quant_args)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

def w_quantize_func(w, quant_args): 
                
    # Asym Linear Quantization
    if quant_args.n_bits_w == 4:
        org_w_shape = w.shape
        q_group_size = quant_args.group_size
        
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        else:
            w = w.reshape(-1, w.shape[-1])

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** quant_args.n_bits_w - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        
        w = (torch.clamp(torch.round(w / scales) +
                        zeros, min_int, max_int) - zeros) * scales
        
        assert torch.isnan(w).sum() == 0

        w_q = w.reshape(org_w_shape)
        
        return w_q.detach()            
    
    if quant_args.n_bits_w == 2:
        
        org_w_shape = w.shape
        q_group_size = quant_args.group_size
        
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        else:
            w = w.reshape(-1, w.shape[-1])

        n = w[0].nelement()
        m = w.data.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1, 1).expand_as(w)
        pos = (w > thres).float()
        neg = (w < -thres).float()
        mask = (w.abs() > thres).float()
        alpha = ((mask * w).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        w = alpha * pos - alpha * neg
        
        assert torch.isnan(w).sum() == 0

        w_q = w.reshape(org_w_shape)

        return w_q.detach()            