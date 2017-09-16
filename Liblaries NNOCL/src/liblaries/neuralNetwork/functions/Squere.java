package liblaries.neuralNetwork.functions;

public class Squere extends Function{
	public Squere(){
		functionID=-126;
		
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
				  + "	if(output[neuron]>0)"
				  + "		output[neuron]*=output[neuron];"
				  + "	else"
				  + "		output[neuron]*=output[neuron]*-1;"
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