package liblaries.neuralNetwork.functions;

public class Sin extends Function{
	public Sin(){
		functionID=-124;
		
		functionOpenCLSource=
				   "float outputFunction(float value) {"
				  + "	if(value>-1.5707963267948966){"
				  + "		if(value<1.5707963267948966)"
				  + "			return sin(output[neuron]);"
				  + "		else return 1;"
				  + "	}else return -1;"
				  + "}";
	}
	public float function(float dana){
		if(dana>-1.5707963267948966){
			if(dana<1.5707963267948966)
				return (float) Math.sin(dana);
			else return 1;
		}
		else return -1;
	}
	/*public float pochodna(float dana){			//This can be wrong
		if(dana>-1){
			if(dana<1)
				return (float) Math.cos(dana);
			else return 1;
		}
		else return -1;
	}*/
}