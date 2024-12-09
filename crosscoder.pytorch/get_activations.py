from dawnet import Inspector
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
crosscoder = ...
inspector = Inspector(model, tokenizer)

# hook the model to the crosscoder operation
# from a feature idx, it means we would like to have the crosscoder already constructed
op_id = inspector.add_op(
    "",
    CrossCoderOp(
        crosscoder=crosscoder,
        layers={"transformer.h.7": lambda x: x[0], "transformer.h.8": lambda x: x[0]},
        name="crosscoder",
    )
)
op = inspector.get_op(op_id)


# run the model: prompt, a feature idx, how much to increase
# difference between updating.
output_ori, state_ori = inspector.run("I love the blue sky")
output, state = inspector.run(
    "I love the blue sky",
    op_params=op.run_params(feature_idx=0, increase=0.1)
)

output2, state2 = inspector.run(
    "I love the blue sky",
    _op_params=op.run_params(feature_idx=0, increase=0.2),
    _state=state,
)

#### generation
inspector.state.clear()
output_ori = inspector._model.generate("I love the blue sky")
state_ori = inspector.state

inspector.op_params = {op_id: op.run_params(feature_idx=0, increase=0.1)}
output = inspector._model.generate("I love the blue sky")
state = inspector.state

inspector.op_params = {op_id: op.run_params(feature_idx=0, increase=0.2)}
output2 = inspector._model.generate("I love the blue sky")
state2 = inspector.state

# cache the model (usually, if you run from output to input, you can efficiently know which one to use cache, which one to run)
