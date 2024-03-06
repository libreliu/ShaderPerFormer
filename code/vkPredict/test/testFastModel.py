import unittest
import torch
from model.custom_roberta import RobertaFastSelfAttention, RobertaFastConfig
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaConfig
import torch.backends.cuda


def compare_gradients(module1, module2, rtol, atol):
    for (param1, param2) in zip(module1.parameters(), module2.parameters()):
        if not torch.allclose(
            param1.grad.to(dtype=torch.float32, device='cpu'),
            param2.grad.to(dtype=torch.float32, device='cpu'),
            rtol, atol):
            return False
    return True

# Ref on mixed precision training:
# - https://blog.csdn.net/kuweicai/article/details/126339584
# However for testing purposes we can just make those specific types only

class FastModelTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_fast_attention(self):
        hidden_size = 16
        num_heads = 2
        seq_len = 5
        batch_size = 2
        device = 'cuda:0'
        refDevice = 'cpu'
        refType = torch.float32

        # torch.backends.cuda.enable_math_sdp(False)

        for tensor_type, attention_type in (
            (torch.float16, "torch-memeff-nomask"),
            (torch.float16, "torch-flash-nomask"),
            (torch.float16, "xformers-memeff"),
            (torch.float16, "xformers-memeff-nomask"),
            (torch.bfloat16, "torch-memeff-nomask"),
            (torch.bfloat16, "torch-flash-nomask"),
            (torch.bfloat16, "xformers-memeff"),
            (torch.bfloat16, "xformers-memeff-nomask"),
            (torch.float32, "xformers-memeff"),
            (torch.float32, "xformers-memeff-nomask")
        ):

            cfg = RobertaConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                attention_probs_dropout_prob=0.5
            )
            cfgFast = RobertaFastConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                attention_probs_dropout_prob=0.5,
                attention_type=attention_type
            )

            origAtten = RobertaSelfAttention(cfg, "absolute").to(device=refDevice, dtype=refType)
            fastAtten = RobertaFastSelfAttention(cfgFast, "absolute").to(device=device, dtype=tensor_type)

            totalTrials = 20
            for trialIdx in range(totalTrials):
                print(f"Running trial {trialIdx + 1} / {totalTrials} for {attention_type} with {tensor_type}")
            
                # initialize weights
                origAtten.query.weight.data.normal_(mean=10.0, std=1.0)
                if origAtten.query.bias is not None:
                    origAtten.query.bias.data.normal_(mean=7.0, std=1.0)
                
                origAtten.key.weight.data.normal_(mean=12.0, std=1.0)
                if origAtten.key.bias is not None:
                    origAtten.key.bias.data.normal_(mean=5.0, std=1.0)

                origAtten.value.weight.data.normal_(mean=15.0, std=1.0)
                if origAtten.value.bias is not None:
                    origAtten.value.bias.data.normal_(mean=6.0, std=1.0)

                # TODO: copy weights in qkv to fastatten
                fastAtten.query.weight.data = origAtten.query.weight.data.clone().to(device=device, dtype=tensor_type)
                if origAtten.query.bias is not None:
                    fastAtten.query.bias.data = origAtten.query.bias.data.clone().to(device=device, dtype=tensor_type)

                fastAtten.key.weight.data = origAtten.key.weight.data.clone().to(device=device, dtype=tensor_type)
                if origAtten.key.bias is not None:
                    fastAtten.key.bias.data = origAtten.key.bias.data.clone().to(device=device, dtype=tensor_type)

                fastAtten.value.weight.data = origAtten.value.weight.data.clone().to(device=device, dtype=tensor_type)
                if origAtten.value.bias is not None:
                    fastAtten.value.bias.data = origAtten.value.bias.data.clone().to(device=device, dtype=tensor_type)

                # seqlen, bsz, hiddendim?
                testBatch = torch.rand((batch_size, seq_len, hidden_size), dtype=tensor_type, device=device)
                testBatchRef = testBatch.clone().to(device=refDevice, dtype=refType)

                origAtten.eval()
                fastAtten.eval()

                origResult = origAtten(testBatchRef)
                fastResult = fastAtten(testBatch)
                
                # print(origResult)
                # print(fastResult)

                rtol = 0.1 if tensor_type is torch.float16 or tensor_type is torch.bfloat16 else 1e-5
                atol = 0.001 if tensor_type is torch.float16 or tensor_type is torch.bfloat16 else 1e-8

                assert(torch.allclose(
                    origResult[0],
                    fastResult[0].to(device=refDevice, dtype=refType),
                    rtol=rtol,
                    atol=atol
                ))

                # a dummy loss
                origLoss = origResult[0].mean()
                fastLoss = fastResult[0].mean()
                
                origLoss.backward()
                fastLoss.backward()

                # print(origResult)
                # print(fastResult)
                # print(fastResult[0].get_device())

                compare_gradients(origAtten, fastAtten, rtol, atol)



