"""Preserve exactly one GAS normalization at the Lightning/DeepSpeed boundary."""
from lightning.pytorch.plugins.precision import DeepSpeedPrecision


class DeepSpeedPrecisionFlow2Act(DeepSpeedPrecision):
    def backward(self,tensor,model,optimizer,*args,**kwargs):
        # Lightning ClosureResult already divided the training loss by GAS.
        # DeepSpeed must not divide that same loss/gradient a second time.
        kwargs["scale_wrt_gas"]=False
        return super().backward(tensor,model,optimizer,*args,**kwargs)
