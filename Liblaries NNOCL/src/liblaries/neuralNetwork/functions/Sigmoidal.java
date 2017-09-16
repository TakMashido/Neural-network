package liblaries.neuralNetwork.functions;

public class Sigmoidal extends Function{
	public Sigmoidal(){
		functionID=-125;
		
		functionKernelSource=
				  "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
				  + "	int neuron=get_global_id(0);"
				  + "	int index=neuron*connectionsNumber;"
				  + "	neuron++;"								//offset of worksize in not supported by openCL yet
				  + "	"
				  + "	output[neuron]=0;"
				  + "	for(int i=0;i<connectionsNumber;i++){"
				  + "		output[neuron]+=weights[index+i]*input[i];"
				  + "	}"
				  + "		"
				  + "		output[neuron]=1/(1+exp(-2*output[neuron]));"
				  + "}";
	}
	public float function(float dana){
		return (float) (1/(1+Math.exp(-2*dana)));
	}
	/*public float pochodna(float dana){
		return dana*(1-dana);
	}*/
}