STAGE_REGISTRY = {}
AUDIT_REGISTRY = {}
HOOK_REGISTRY = {}
EVAL_REGISTRY = {}

def register(registry, name):
    def deco(cls):
        registry[name] = cls
        return cls
    return deco
