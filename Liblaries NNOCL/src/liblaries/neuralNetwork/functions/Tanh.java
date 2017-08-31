package liblaries.neuralNetwork.functions;

public class Tanh extends Function{
	public Tanh(){
		functionID=-127;
		
		functionKernelSource=
				  "__kernel void sumOutput(__global float *preWyjscia,__global float *wyjscia, int liczbaPolaczen){"
				+ "		int neuron=get_global_id(0);"
				+ "		int index=neuron*liczbaPolaczen;"
				+ "		"
				+ "		wyjscia[neuron]=0;"
				+ "		for(int i=0;i<liczbaPolaczen;i++){"
				+ "			wyjscia[neuron]+=preWyjscia[index+i];"
				+ "		}"
				+ "		wyjscia[neuron]=tanh(wyjscia[neuron]);"
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