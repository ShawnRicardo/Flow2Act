"""Bound native ZeRO flatten temporaries without splitting a parameter tensor."""
from typing import Iterable


def native_optimizer_groups_flow2act(named_parameters: Iterable, *, weight_decay=0.1,
                                            max_group_elements=0):
    if type(max_group_elements) is not int or max_group_elements < 0:
        raise ValueError("max_optimizer_group_elements must be a nonnegative integer")
    named_parameters=list(named_parameters)
    no_decay=("bias","norm","Norm","ln","Ln","rotary_emb","adaln")
    groups=[];descriptions=[]
    for source_group,decay in enumerate((True,False)):
        selected=[(name,p) for name,p in named_parameters if p.requires_grad and
                  (not any(term in name.lower() for term in no_decay))==decay]
        chunks=[];current=[];count=0
        for name,p in selected:
            if max_group_elements and current and count+p.numel()>max_group_elements:
                chunks.append(current);current=[];count=0
            current.append((name,p));count+=p.numel()
        # Keep the native empty source-group behavior when there are no tensors.
        chunks.append(current)
        for chunk in chunks:
            decay_value=weight_decay if decay else 0.0
            groups.append({"params":[p for _,p in chunk],"weight_decay":decay_value})
            descriptions.append({"source_group":source_group,"weight_decay":float(decay_value),
                "parameters":[{"name":name,"shape":list(p.shape)} for name,p in chunk]})
    return groups,{"max_optimizer_group_elements":max_group_elements,"groups":descriptions}


def validate_optimizer_groups_flow2act(expected, saved):
    if expected is None and saved is None:
        return
    if saved is None and expected is not None and expected["max_optimizer_group_elements"]==0:
        import warnings
        warnings.warn("Legacy checkpoint has no optimizer grouping metadata; accepting only the native cap=0 grouping",
                      UserWarning,stacklevel=2)
        return
    if expected != saved:
        raise ValueError("Checkpoint optimizer grouping contract mismatch")
