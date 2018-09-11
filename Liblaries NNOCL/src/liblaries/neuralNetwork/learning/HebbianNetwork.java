package liblaries.neuralNetwork.learning;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import liblaries.neuralNetwork.functions.Function;

public class HebbianNetwork extends LNetwork{
	public HebbianNetwork(){}
	public HebbianNetwork(float[][][] weights, Function function) {
		super(weights,function);
	}
	public HebbianNetwork(float[][][] weights, Function function, boolean initializeOpenCL) {
		super(weights,function,initializeOpenCL);
	}
	public HebbianNetwork(int inputsNumber,int[] layersSize, Function function) {
		super(inputsNumber,layersSize,function);
	}
	public HebbianNetwork(int inputsNumber,int[] layersSize, Function function, boolean initializeOpenCL) {
		super(inputsNumber,layersSize,function,initializeOpenCL);
	}
	
	protected void addOpenCLProgram() {
		super.addOpenCLProgram();
		openCLProgram.put("calculateWeights",
				   "__kernel void calculateWeights(__global float *weights,__global float *deltaWeights,__global float *output,__global float *input,int connectionsNumber, float n, float m){"
				 + "	int neuron=get_global_id(0)+1;"							//Offset(bias)
				 + "	int index=neuron*connectionsNumber;"
				 + "	"
				 + "	float outputValue=output[neuron];"
				 + "	index++;"
				 + "	for(int i=1;i<connectionsNumber;i++){"
				 + "		float delta=n*input[i]*outputValue+m*deltaWeights[index];"
				 + "		weights[index]=weights[index]+delta;"
				 + "		weightsSum+=weights[index];"
				 + "		deltaWeights[index]=delta;"
				 + "		index++;"
				 + "	}"
				 + "}");
	}
	
	public void countWeights(int elementNr, float n, float m) {
		if(openCLLoaded) {
			CL.clSetKernelArg(calculateWeightsKernel, 5, Sizeof.cl_float, Pointer.to(new float[] {n}));
			CL.clSetKernelArg(calculateWeightsKernel, 6, Sizeof.cl_float, Pointer.to(new float[] {m}));
			for(int i=layersNumber-1;i>-1;i--) {
				CL.clSetKernelArg(calculateWeightsKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 1, Sizeof.cl_mem, Pointer.to(deltaWeightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 2, Sizeof.cl_mem, Pointer.to(outputsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 3, Sizeof.cl_mem, Pointer.to(i==0?learningSequence[i].inputsCL:outputsCL[i-1]));
				CL.clSetKernelArg(calculateWeightsKernel, 4, Sizeof.cl_int, Pointer.to(new int[] {i==0?learningSequence[i].inputs.length:layersSize[i-1]}));
				CL.clEnqueueNDRangeKernel(commandQueue, calculateWeightsKernel, 1, null, new long[] {layersSize[i]+1}, new long[] {1}, 0, null, null);
			}
		}else {
			for(int i=0;i<weights.length;i++){
				for(int j=0;j<weights[i].length;j++) {
					for(int k=1;k<weights[i][j].length;k++){
						//float delta=m*deltaWeights[i][j][k]+n*((k==0?1:(i==0?learningSequence[elementNr].inputs[k-1]:output[i-1][k-1]))-weights[i][j][k]);
						float delta=m*deltaWeights[i][j][k]+n*(i==0?learningSequence[elementNr].inputs[k-1]:output[i-1][k-1])*output[i][j];
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
				}
			}
		}
	}
}