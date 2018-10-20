package liblaries.neuralNetwork.learning;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import liblaries.neuralNetwork.functions.OutputFunction;

public class BackPropagationNetwork extends LNetwork{
	public BackPropagationNetwork(){}
	public BackPropagationNetwork(float[][][] weights, OutputFunction outputFunction) {
		super(weights,outputFunction);
	}
	public BackPropagationNetwork(float[][][] weights, OutputFunction outputFunction, boolean initializeOpenCL) {
		super(weights,outputFunction,initializeOpenCL);
	}
	public BackPropagationNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction) {
		super(inputsNumber, layersSize,outputFunction);
	}
	public BackPropagationNetwork(int inputsNumber,int[] layersSize, OutputFunction outputFunction, boolean initializeOpenCL) {
		super(inputsNumber,layersSize,outputFunction,initializeOpenCL);
	}
	protected void prepareData() {
		super.prepareData();
		prepareError();
	}
	protected void addOpenCLProgram() {
		super.addOpenCLProgram();
		openCLProgram.put("outputError",
				   "__kernel void outputError(__global float *error, __global float *outputs,__global float *goodOutputs){"
				 + "	int neuron=get_global_id(0);"
				 + "	error[neuron]=goodOutputs[neuron]-outputs[neuron+1];"
				 + "}");
		openCLProgram.put("calculateError",
				   "__kernel void calculateError(__global const float *weights, __global const float *errorUp, __global float *error, int connectionsNumber){"
				 + "	int neuron=get_global_id(0);"
				 + "	int neuron1=neuron+1;"
				 + "	"
				 + "	error[neuron]=0;"
				 + "	for(int i=0;i<connectionsNumber;i++){"
				 + "		error[neuron]+=errorUp[i]*weights[neuron1+i*connectionsNumber];"
				 + "	}"
				 + "}");
		openCLProgram.put("calculateWeights",
				   "__kernel void calculateWeights(__global float *weights,__global float *deltaWeights,__global float *error,__global float *input,int connectionsNumber, float n, float m){"
				 + "	int neuron=get_global_id(0);"
				 + "	int connection=get_global_id(1);"
				 + "	int index=neuron*connectionsNumber+connection;"
				 + "	"
				 + "	float delta=n*input[connection]*error[neuron]+m*deltaWeights[index];"						//TODO SNOCL find where put the derivative of a function
				 + "	weights[index]+=delta;"
				 + "	deltaWeights[index]=delta;"
				 + "}");
	}
	
 	public void countError(int elementNr){
		boolean pom=true;
		
		if(openCLLoaded) {
			CL.clSetKernelArg(outputLayerErrorKernel, 0, Sizeof.cl_mem, Pointer.to(errorCL[layersNumber-1]));			//Input layer
			CL.clSetKernelArg(outputLayerErrorKernel, 1, Sizeof.cl_mem, Pointer.to(outputsCL[layersNumber-1]));
			CL.clSetKernelArg(outputLayerErrorKernel, 2, Sizeof.cl_mem, Pointer.to(learningSequence[elementNr].outputsCL));
			CL.clEnqueueNDRangeKernel(commandQueue, outputLayerErrorKernel, 1, null, new long[] {layersSize[layersNumber-1]}, new long[] {1}, 0, null, null);
			
			for(int nrLayer=layersNumber-2;nrLayer>-1;nrLayer--){													//Hidden layer
				System.out.println("check this");
				int neurons=layersSize[nrLayer+1];			//WTF? this work
				int connections=layersSize[nrLayer+1];
				
				CL.clSetKernelArg(calculateErrorKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer+1]));
				CL.clSetKernelArg(calculateErrorKernel, 1, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer+1]));
				CL.clSetKernelArg(calculateErrorKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[nrLayer]));
				CL.clSetKernelArg(calculateErrorKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections+1}));
				CL.clEnqueueNDRangeKernel(commandQueue, calculateErrorKernel, 1, null, new long[] {neurons}, new long[] {1}, 0, null, null);
			}
		}else {
			for(int nrLayer=weights.length-1;nrLayer>-1;nrLayer--){
				if(pom){																							//Input layer
					for(int i=0;i<weights[nrLayer].length;i++){
						error[nrLayer][i]=learningSequence[elementNr].outputs[i]-outputs[nrLayer][i];				//Dont't have to reset error
					}
					pom=false;
				}else{																								//Hidden layer
					for(int i=0;i<weights[nrLayer].length;i++)														//Reset error
						error[nrLayer][i]=0;
					
					for(int i=0;i<weights[nrLayer+1].length;i++){
						for(int j=0;j<weights[nrLayer+1][i].length-1;j++){
							error[nrLayer][j]+=error[nrLayer+1][i]*weights[nrLayer+1][i][j+1];
						}
					}
				}
			}
		}
	}
	public void lCountWeights(int elementNr,float n,float m){
		countError(elementNr);
		
		if(openCLLoaded) {
			//long time=System.nanoTime();
			CL.clSetKernelArg(calculateWeightsKernel, 5, Sizeof.cl_float, Pointer.to(new float[] {n}));
			CL.clSetKernelArg(calculateWeightsKernel, 6, Sizeof.cl_float, Pointer.to(new float[] {m}));
			for(int i=layersNumber-1;i>-1;i--) {
				//long time=System.nanoTime();
				CL.clSetKernelArg(calculateWeightsKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 1, Sizeof.cl_mem, Pointer.to(deltaWeightsCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 2, Sizeof.cl_mem, Pointer.to(errorCL[i]));
				CL.clSetKernelArg(calculateWeightsKernel, 3, Sizeof.cl_mem, Pointer.to(i==0?learningSequence[i].inputsCL:outputsCL[i-1]));
				CL.clSetKernelArg(calculateWeightsKernel, 4, Sizeof.cl_int, Pointer.to(new int[] {layersSize[i]+1}));
				//System.out.println("settingup time="+((System.nanoTime()-time)/1000)/1000f+"ms");
				//time=System.nanoTime();
				CL.clEnqueueNDRangeKernel(commandQueue, calculateWeightsKernel, 2, null, new long[] {layersSize[i+1],layersSize[i]+1}, new long[] {1,1}, 0, null, null);
				//System.out.println("invoking time="+((System.nanoTime()-time)/1000)/1000f+"ms");
			}
			//System.out.println("time="+((System.nanoTime()-time)/1000)/1000f+"ms");
		}else {
			//long time=System.nanoTime();
			for(int i=0;i<weights.length;i++){
				for(int j=0;j<weights[i].length;j++){
					for(int k=0;k<weights[i][j].length;k++){
						float delta=m*deltaWeights[i][j][k]+n*error[i][j]*(k==0?1:(i==0?learningSequence[elementNr].inputs[k-1]:outputs[i-1][k-1]));					//FIXME NN find where put derivative of function
						
						weights[i][j][k]+=delta;
						deltaWeights[i][j][k]=delta;
					}
				}
			}
			//System.out.println("time="+((System.nanoTime()-time)/1000)/1000f+"ms");
		}
	}
}