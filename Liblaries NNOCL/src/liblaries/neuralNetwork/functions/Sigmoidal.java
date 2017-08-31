package liblaries.neuralNetwork.functions;

public class Sigmoidal extends Function{
	public Sigmoidal(){
		functionID=-125;
		
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
				+ "		float pom=0;"
				+ "		wyjscia[neuron]=1/(1+exp(-2*wyjscia[neuron]));"
				+ "}";
	}
	public float function(float dana){
		return (float) (1/(1+Math.exp(-2*dana)));
	}
	/*public float pochodna(float dana){
		return dana*(1-dana);
	}*/
}