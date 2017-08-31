package liblaries.neuralNetwork.functions;

public class Squere extends Function{
	public Squere(){
		functionID=-126;
		
		functionKernelSource=											//TODO SNJOCL sprawdziæ
				  "__kernel void sumOutput(__global float *preWyjscia,__global float *wyjscia, int liczbaPolaczen){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*liczbaPolaczen;"
				+ "		"
				+ "		wyjscia[neuron]=0;"
				+ "		for(int i=0;i<liczbaPolaczen;i++){"
				+ "			wyjscia[neuron]+=preWyjscia[index+i];"
				+ "		}"
				+ "		if(wyjscia[neuron]>0)"
				+ "			wyjscia[neuron]*=wyjscia[neuron];"
				+ "		else"
				+ "			wyjscia[neuron]*=wyjscia[neuron]*-1;"
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