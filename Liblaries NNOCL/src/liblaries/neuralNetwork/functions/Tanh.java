package liblaries.neuralNetwork.functions;

public class Tanh extends Function{
	public Tanh(){
		functionID=-127;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preOutput,__global float *output, int connectionNumber){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*connectionNumber;"
				+ "		"
				+ "		output[neuron]=0;"
				+ "		for(int i=0;i<connectionNumber;i++){"
				+ "			output[neuron]+=preOutput[index+i];"
				+ "		}"
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