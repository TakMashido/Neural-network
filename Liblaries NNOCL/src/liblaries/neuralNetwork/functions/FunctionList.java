package liblaries.neuralNetwork.functions;

public class FunctionList{
	public static final byte linearID=-128;
	public static final byte tanhID=-127;
	public static final byte squereID=-126;
	public static final byte LogistycznaID=-125;
	public static final byte SinID=-124;
	
	public static OutputFunction getFunction(byte IDFunkcji){
		switch(IDFunkcji){
		case linearID:return new Linear();
		case tanhID:return new Tanh();
		case squereID:return new Squere();
		case LogistycznaID:return new Sigmoidal();
		case SinID:return new Sin();
		}
		throw new IllegalArgumentException();
	}
	public static OutputFunction geFunction(String args) {
		switch(args) {
		case "linear":		return new Linear();
		case "tanh":		return new Tanh();
		case "squere":		return new Squere();
		case "sigmoidal":	return new Sigmoidal();
		case "sin":			return new Sin();
		}
		return null;
	}
}