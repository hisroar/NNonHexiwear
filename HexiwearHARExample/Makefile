MAKE=make
MBED=mbed
ECHO=@echo
UTENSOR=utensor-cli
TARGET=hexiwear
PROFILE=uTensor/build_profile/release.json
NN_PATH=output/
NN=dnn6_2
OUTPUT_NODE=y_pred
BUILD_DIR=BUILD/
MODELS_DIR=models/
define EVALCODE_BODY
////////////////////////////////////////////////////////////////////////////////
Context ctx;  //creating the context class, the stage where inferences take place 
// wrapping the input data in a tensor class
Tensor* input_x = new WrappedRamTensor<float>({1, [INPUT_DATA_SIZE]}, (float*) input_data);

get_$(NN)_ctx(ctx, input_x);  // pass the tensor to the context
S_TENSOR pred_tensor = ctx.get("y_pred:0");  // getting a reference to the output tensor
ctx.eval(); //trigger the inference

int pred_label = *(pred_tensor->read<int>(0, 0));  //getting the result back
printf("Predicted label: %d\n\n", pred_label);
////////////////////////////////////////////////////////////////////////////////
endef
export EVALCODE_BODY

all:
	$(MAKE) models && $(MAKE) compile

compile: main.cpp
	$(MBED) compile -m $(TARGET) -t GCC_ARM --profile=$(PROFILE)

models: $(NN_PATH)$(NN).pb
	$(UTENSOR) convert $(NN_PATH)$(NN).pb --output-nodes=$(OUTPUT_NODE)
	$(ECHO) "\n\n*** Make sure to add '#include \"$(MODELS_DIR)$(NN).hpp\"' to main.cpp ***\n\n"

evalcode:
	$(ECHO) "\n*** Make sure to add '#include \"$(MODELS_DIR)$(NN).hpp\"' to main.cpp ***\n"
	$(ECHO) "Add the following code to main.cpp to evaluate neural network given input"
	$(ECHO) "$$EVALCODE_BODY"

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MODELS_DIR)*

.PHONY: all compile models cleans
