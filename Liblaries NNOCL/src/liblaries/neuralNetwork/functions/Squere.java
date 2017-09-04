package liblaries.neuralNetwork.functions;

public class Squere extends Function{
	public Squere(){
		functionID=-126;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preOutput,__global float *output, int connectionNumber){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*connectionNumber;"
				+ "		"
				+ "		output[neuron]=0;"
				+ "		for(int i=0;i<connectionNumber;i++){"
				+ "			output[neuron]+=preOutput[index+i];"
				+ "		}"
				+ "		if(output[neuron]>0)"
				+ "			output[neuron]*=output[neuron];"
				+ "		else"
				+ "			output[neuron]*=output[neuron]*-1;"
				+ "}";
	}
	public float function(float dana){
		if(dana<1&&dana>-1){
			if(dana<0){
				return -dana*dana;
			}else{
				return dana*dana;
			}
		}
		return dana;
	}
	/*public float pochodna(float dana){
		if(dana<1&&dana>-1){
			return 2*dana;
		}
		return dana;
	}*/
}