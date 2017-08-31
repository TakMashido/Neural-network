package liblaries.neuralNetwork.functions;

public class FunctionList{
	public static final byte LiniowaID=-128;
	public static final byte TangensHiperbolicznyID=-127;
	public static final byte KwadratowaID=-126;
	public static final byte LogistycznaID=-125;
	public static final byte SinusID=-124;
	
	public static Function getFunction(byte IDFunkcji){
		switch(IDFunkcji){
		case LiniowaID:return new Linear();
		case TangensHiperbolicznyID:return new Tanh();
		case KwadratowaID:return new Squere();
		case LogistycznaID:return new Sigmoidal();
		case SinusID:return new Sin();
		}
		throw new IllegalArgumentException();
	}
}