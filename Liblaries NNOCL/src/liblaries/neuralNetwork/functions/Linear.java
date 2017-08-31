package liblaries.neuralNetwork.functions;

public class Linear extends Function{
	public Linear() {
		functionID=-128;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preOutput,__global float *output, int connectionsNumber){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*connectionsNumber;"
				+ "		"
				+ "		output[neuron]=0;"
				+ "		for(int i=0;i<connectionsNumber;i++){"
				+ "			output[neuron]+=preOutput[index+i];"
				+ "		}"
				+ "}";
	}

	public float function(float dana) {
		return dana;
	}
}