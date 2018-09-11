package liblaries.neuralNetwork.functions;

public class Squere extends Function{
	public Squere(){
		functionID=-126;
		
		functionOpenCLSource=
				   "float outputFunction(float value) {"
				  + "	if(value>0)"
				  + "		return value*value;"
				  + "	else"
				  + "		return value*value*-1;"
				  + "}";
	}
	public float function(float value){
		if(value<1&&value>-1){
			if(value<0){
				return -value*value;
			}else{
				return value*value;
			}
		}
		return value;
	}
	/*public float pochodna(float dana){
		if(dana<1&&dana>-1){
			return 2*dana;
		}
		return dana;
	}*/
}