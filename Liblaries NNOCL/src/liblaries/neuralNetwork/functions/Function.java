package liblaries.neuralNetwork.functions;

public abstract class Function{
	protected byte functionID;
	protected String functionOpenCLSource;
	
	public abstract float function(float dana);
	//public abstract float pochodna(float dana);

	public final String getOpenCLProgram() {
		return functionOpenCLSource;
	}

	public final byte getFunctionID() {
		return functionID;
	}
}