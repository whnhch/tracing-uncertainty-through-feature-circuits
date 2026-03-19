from torch import nn

class PatchedMLP(nn.Module):
    def __init__(self, transcoder):
        super().__init__()
        self.transcoder = transcoder

    def forward(self, x):
        # 1. Save the original dtype the LLM is using (BFloat16)
        original_dtype = x.dtype

        # 2. Find out what dtype the transcoder is using (Float32)
        transcoder_dtype = next(self.transcoder.parameters()).dtype

        # 3. Cast the input to match the transcoder
        x_cast = x.to(transcoder_dtype)

        # 4. Run the transcoder
        x_out_pred, _ = self.transcoder(x_cast)

        # 5. Cast the output back to the LLM's original dtype before returning
        return x_out_pred.to(original_dtype)