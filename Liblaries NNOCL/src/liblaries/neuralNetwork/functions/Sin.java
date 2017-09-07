package liblaries.neuralNetwork.functions;

public class Sin extends Function{
	public Sin(){
		functionID=-124;
		
		functionKernelSource=
				  "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
				  + "	int neuron=get_global_id(0);"
				  + "	int index=neuron*connectionsNumber;"
				  + "	"
				  + "	output[neuron]=0;"
				  + "	for(int i=0;i<connectionsNumber;i++){"
				  + "		output[neuron]+=weights[index+i]*input[i];"
				  + "	}"
				  + "		"
				  + "		if(output[neuron]>-1.5707963267948966){"
				  + "			if(output[neuron]<1.5707963267948966)"
				  + "				output[neuron]=sin(output[neuron]);"
				  + "			else output[neuron]=1;"
				  + "		}else output[neuron]=-1;"
				  + "}";
	}
	public float function(float dana){
		if(dana>-1.5707963267948966){
			if(dana<1.5707963267948966)
				return (float) Math.sin(dana);
			else return 1;
		}
		else return -1;
	}
	/*public float pochodna(float dana){			//TODO SNOCL upewniæ siê czy pochodna dobra
		if(dana>-1){
			if(dana<1)
				return (float) Math.cos(dana);
			else return 1;
		}
		else return -1;
	}*/
}