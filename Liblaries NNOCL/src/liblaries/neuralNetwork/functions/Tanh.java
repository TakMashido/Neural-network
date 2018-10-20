package liblaries.neuralNetwork.functions;

public class Tanh extends OutputFunction{
	public Tanh(){
		functionID=-127;
		
		functionOpenCLSource=
					"float outputFunction(float value) {"
				  + "	return tanh(value);"
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