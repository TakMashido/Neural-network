package liblaries.neuralNetwork.functions;

public class Linear extends Function{
	public Linear() {
		functionID=-128;
		
		functionKernelSource=
				  "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
				  + "	int neuron=get_global_id(0);"
				  + "	int index=neuron*connectionsNumber;"
				  + "	"
				  + "	output[neuron]=0;"
				  + "	for(int i=0;i<connectionsNumber;i++){"
				  + "		output[neuron]+=weights[index+i]*input[i];"
				  + "	}"
				  + "}";
	}

	public float function(float dana) {
		return dana;
	}
}