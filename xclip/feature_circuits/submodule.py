from dataclasses import dataclass

import torch as t
from einops import rearrange
from nnsight.envoy import Envoy


@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Envoy
    use_input: bool = False
    is_tuple: bool = False
    shape: tuple = None

    def __hash__(self):
        return hash(self.name)

    def get_activation(self):
        if self.use_input:
            # out = self.submodule.inputs[0][0]
            out = self.submodule.input
        else:
            if self.is_tuple:
                out = self.submodule.output[0]
            else:
                out = self.submodule.output

        # trick to handle data that's not in the format batch x seq_len x features
        if self.shape is not None:
            if len(self.shape) == 2:
                out = rearrange(out, 'b c -> b 1 c')
            elif len(self.shape) == 3:
                pass
            elif len(self.shape) == 4:
                out = rearrange(out, 'b c h w -> b (h w) c')
            else:
                raise NotImplementedError

        return out

    def set_activation(self, x):
        # trick to handle data that's not in the format batch x seq_len x features
        if self.shape is not None:
            if len(self.shape) == 2:
                x = rearrange(x, 'b 1 c -> b c', c=self.shape[1])
            elif len(self.shape) == 3:
                pass
            elif len(self.shape) == 4:
                x = rearrange(x, 'b (h w) c -> b c h w', h=self.shape[2], w=self.shape[3], c=self.shape[1])
            else:
                raise NotImplementedError

        if self.use_input:
            # self.submodule.inputs[0][0][:] = x
            self.submodule.input[:] = x
        else:
            if self.is_tuple:
                self.submodule.output[0][:] = x
            else:
                self.submodule.output = x

    def stop_grad(self):
        if self.use_input:
            # self.submodule.inputs[0][0].grad = t.zeros_like(self.submodule.inputs[0][0])
            self.submodule.input.grad = t.zeros_like(self.submodule.input)
        else:
            if self.is_tuple:
                self.submodule.output[0].grad = t.zeros_like(self.submodule.output[0])
            else:
                self.submodule.output.grad = t.zeros_like(self.submodule.output)
