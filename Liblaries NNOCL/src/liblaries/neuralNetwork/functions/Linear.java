package liblaries.neuralNetwork.functions;

public class Linear extends Function{
	public Linear() {
		functionID=-128;
		
		functionOpenCLSource=
				    "float outputFunction(float value) {"
				  + "	return value;"
				  + "}";
	}

	public float function(float dana) {
		return dana;
	}
}