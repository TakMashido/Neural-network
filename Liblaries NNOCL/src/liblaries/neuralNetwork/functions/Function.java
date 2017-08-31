package liblaries.neuralNetwork.functions;

public abstract class Function{
	protected byte functionID;
	protected String functionKernelSource;
	
	public abstract float function(float dana);
	//public abstract float pochodna(float dana);

	public final String getOpenCLProgram() {
		return functionKernelSource;
	}

	public byte getFunctionID() {
		return functionID;
	}
}