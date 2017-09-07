package liblaries.neuralNetwork.functions;

public class Tanh extends Function{
	public Tanh(){
		functionID=-127;
		
		functionKernelSource=
				  "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
				  + "	int neuron=get_global_id(0);"
				  + "	int index=neuron*connectionsNumber;"
				  + "	"
				  + "	output[neuron]=0;"
				  + "	for(int i=0;i<connectionsNumber;i++){"
				  + "		output[neuron]+=weights[index+i]*input[i];"
				  + "	}"
				  + "	"
				  + "		output[neuron]=tanh(output[neuron]);"
				  + "}";
	}
	public float function(float dana){
		return (float) Math.tanh(dana);
	}
	/*public float pochodna(float dana){
		return (1+dana)*(1-dana);
		//return dana;
	}*/
}