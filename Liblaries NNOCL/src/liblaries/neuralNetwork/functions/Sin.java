package liblaries.neuralNetwork.functions;

public class Sin extends Function{
	public Sin(){
		functionID=-124;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preWyjscia,__global float *wyjscia, int liczbaPolaczen){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*liczbaPolaczen;"
				+ "		"
				+ "		wyjscia[neuron]=0;"
				+ "		for(int i=0;i<liczbaPolaczen;i++){"
				+ "			wyjscia[neuron]+=preWyjscia[index+i];"
				+ "		}"
				+ "		"
				+ "		if(wyjscia[neuron]>-1.5707963267948966){"
				+ "			if(wyjscia[neuron]<1.5707963267948966)"
				+ "				wyjscia[neuron]=sin(wyjscia[neuron]);"
				+ "			else wyjscia[neuron]=1;"
				+ "		}else wyjscia[neuron]=-1;"
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