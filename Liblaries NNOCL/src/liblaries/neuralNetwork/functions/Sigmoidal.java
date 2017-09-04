package liblaries.neuralNetwork.functions;

public class Sigmoidal extends Function{
	public Sigmoidal(){
		functionID=-125;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preOutput,__global float *output, int connectionNumber){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*connectionNumber;"
				+ "		"
				+ "		output[neuron]=0;"
				+ "		for(int i=0;i<connectionNumber;i++){"
				+ "			output[neuron]+=preOutput[index+i];"
				+ "		}"
				+ "		"
				+ "		float pom=0;"
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