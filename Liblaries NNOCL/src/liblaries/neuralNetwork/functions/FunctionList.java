package liblaries.neuralNetwork.functions;

public class FunctionList{
	public static final byte linearID=-128;
	public static final byte tanhID=-127;
	public static final byte KwadratowaID=-126;
	public static final byte LogistycznaID=-125;
	public static final byte SinID=-124;
	
	public static Function getFunction(byte IDFunkcji){
		switch(IDFunkcji){
		case linearID:return new Linear();
		case tanhID:return new Tanh();
		case KwadratowaID:return new Squere();
		case LogistycznaID:return new Sigmoidal();
		case SinID:return new Sin();
		}
		throw new IllegalArgumentException();
	}
}