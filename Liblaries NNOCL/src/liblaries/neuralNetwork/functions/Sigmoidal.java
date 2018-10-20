package liblaries.neuralNetwork.functions;

public class Sigmoidal extends OutputFunction{
	public Sigmoidal(){
		functionID=-125;
		
		functionOpenCLSource=
				    "float outputFunction(float value) {"
				  + "	return 1/(1+exp(-2*value));"
				  + "}";
	}
	public float function(float dana){
		return (float) (1/(1+Math.exp(-2*dana)));
	}
	/*public float pochodna(float dana){
		return dana*(1-dana);
	}*/
}