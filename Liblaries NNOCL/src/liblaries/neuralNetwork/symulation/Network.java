package liblaries.neuralNetwork.symulation;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import liblaries.neuralNetwork.errors.NeuralException;
import liblaries.neuralNetwork.functions.OutputFunction;
import liblaries.neuralNetwork.learning.LNetwork;

public class Network{
	public SymulationFunction[] symulationFunctions=new SymulationFunction[] {
		new SymulationFunction() {
			public void apply(float[] inputData) {
				for(int nrLayer=0;nrLayer<weights.length;nrLayer++){
					if(nrLayer!=0){
						inputData=outputs[nrLayer-1];
					}
					
					for(int i=0;i<weights[nrLayer].length;i++) {
						outputs[nrLayer][i]=weights[nrLayer][i][0];
						for(int j=1;j<weights[nrLayer][i].length;j++){
							outputs[nrLayer][i]+=weights[nrLayer][i][j]*inputData[j-1];
						}
						
						outputs[nrLayer][i]=outputFunction.function(outputs[nrLayer][i]);
					}
				}
			}
			public void applyOpenCL(float[] inputData) {
				cl_mem inputDataCL;
				for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
					if(nrLayer==0){																//Input layer
						inputDataCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(inputData.length+1), Pointer.to(inputData), null);
						CL.clEnqueueWriteBuffer(commandQueue, inputDataCL, CL.CL_TRUE, 0, Sizeof.cl_float, Pointer.to(new float[] {1}), 0, null, null);
						CL.clEnqueueWriteBuffer(commandQueue, inputDataCL, CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*inputNumber, Pointer.to(inputData), 0, null, null);
					}else{
						inputDataCL=outputsCL[nrLayer-1];
					}
					int neurons=layersSize[nrLayer+1];
					int connections=layersSize[nrLayer]+1;
					
					CL.clSetKernelArg(simulateKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer]));
					CL.clSetKernelArg(simulateKernel, 1, Sizeof.cl_mem, Pointer.to(inputDataCL));
					CL.clSetKernelArg(simulateKernel, 2, Sizeof.cl_mem, Pointer.to(outputsCL[nrLayer]));
					CL.clSetKernelArg(simulateKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
					CL.clEnqueueNDRangeKernel(commandQueue, simulateKernel, 1, null ,new long[] {neurons}, new long[]{1}, 0, null, null);
				}
			}
		},
		new SymulationFunction() {
			public void prepareData(float[] inputData) {
				float sum=inputData[0]*inputData[0];
				for(int i=1;i<inputData.length;i++) {
					sum+=inputData[i]*inputData[i];
				}
				sum=(float)Math.sqrt(sum);
				for(int i=0;i<inputData.length;i++) {
					inputData[i]/=sum;
				}
			}
			public void apply(float[] inputData) {
				//prepareData(inputData);
				for(int nrLayer=0;nrLayer<weights.length;nrLayer++){
					if(nrLayer!=0){
						inputData=outputs[nrLayer-1];
					}
					
					for(int i=0;i<weights[nrLayer].length;i++) {
						outputs[nrLayer][i]=weights[nrLayer][i][0];
						for(int j=1;j<weights[nrLayer][i].length;j++){
							float delta=inputData[j-1]-weights[nrLayer][i][j];
							outputs[nrLayer][i]+=delta*delta;
						}
						outputs[nrLayer][i]=-(float)Math.sqrt(outputs[nrLayer][i]);
						
						//outputs[nrLayer][i]=outputFunction.function(outputs[nrLayer][i]);
					}
				}
			}
		}
	};
	
	private float[][][] weights;				//Input layer ID=0			[a][b][c] a->layer, b->neuron, c->connection		//First connection in each neuron is bias
	private float[][] outputs;
	
	private int inputNumber;
	
	public OutputFunction outputFunction;
	
	int[] layersSize;
	int layersNumber;
	
	private boolean openCLLoaded=false;
	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private cl_kernel simulateKernel;
	
	private cl_mem[] weightsCL;							//[a] a->layer
	private cl_mem[] outputsCL;							//[a]
	
	public SymulationFunction symFunction=symulationFunctions[0];
	
	//public Network() {}			//to be or not to be?
	public Network(LNetwork origin) {
		this(origin.getWeights(),origin.getFunction());
		symFunction=symulationFunctions[origin.getSimMethodID()];
	}
	public Network(float[][][] weights, OutputFunction outputFunction) {
		setWeights(weights);
		this.outputFunction=outputFunction;
	}
	public Network(float[][][] weights, OutputFunction outputFunction, byte symulationMethodID) {
		this(weights,outputFunction);
		symFunction=symulationFunctions[symulationMethodID];
	}
	public Network(float[][][] weights, OutputFunction outputFunction, boolean initializeOpenCL) {
		setWeights(weights);
		this.outputFunction=outputFunction;
		
		if(initializeOpenCL)
			initializeOpenCL();
	}
	public void setWeights(float[][][] weights) {
		inputNumber=weights[0][0].length-1;
		this.weights=weights;
		
		if(openCLLoaded)
			createCLMem();
		
		prepareData();
	}
	
	public void initializeOpenCL() {
		CL.setExceptionsEnabled(true);
		
		int[] platformNumber=new int[1];
		CL.clGetPlatformIDs(1, null, platformNumber);
		cl_platform_id platforms[]=new cl_platform_id[platformNumber[0]];
		CL.clGetPlatformIDs(platformNumber[0], platforms, null);
		
		int[] deviceNumber=new int[1];
		CL.clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_ALL, 1, null, deviceNumber);
		cl_device_id[] devices=new cl_device_id[deviceNumber[0]];
		CL.clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_ALL, deviceNumber[0], devices, null);
		
		cl_context_properties contextProperties=new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platforms[0]);
		
		context=CL.clCreateContext(contextProperties, 1, new cl_device_id[] {devices[0]}, null, null, null);
		
		cl_queue_properties queueProperties=new cl_queue_properties();
		commandQueue=CL.clCreateCommandQueueWithProperties(context, devices[0], queueProperties, null);
		
		program=CL.clCreateProgramWithSource(context, 2, new String[] {outputFunction.getOpenCLProgram(),openCLProgram}, null, null);
		CL.clBuildProgram(program, 1, new cl_device_id[] {devices[0]}, null, null, null);
		
		simulateKernel=CL.clCreateKernel(program, "simulate", null);
				
		createCLMem();
		
		openCLLoaded=true;
	}
	public void clearOpenCL() {
		openCLLoaded=false;
		
		for(int i=0;i<weights.length;i++) {
			CL.clReleaseMemObject(weightsCL[i]);
			CL.clReleaseMemObject(outputsCL[i]);
		}
		
		//CL.clReleaseKernel(countOutputsKernel);
		CL.clReleaseKernel(simulateKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
	}
	private void createCLMem() {
		weightsCL=new cl_mem[weights.length];
		outputsCL=new cl_mem[weights.length];
		
		float[] wagiCLSrc;
		
		int index;
		for(int i=0;i<weights.length;i++) {
			int liczbaPo³¹czeñ=layersSize[i]+1;
			wagiCLSrc=new float[weights[i].length*liczbaPo³¹czeñ];
			
			index=0;
			for(int j=0;j<weights[i].length;j++) {
				for(int k=0;k<weights[i][j].length;k++) {
					wagiCLSrc[index]=weights[i][j][k];
					index++;
				}
			}
			weightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(index+1), Pointer.to(wagiCLSrc), null);
			
			outputsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(outputs[i].length+1), Pointer.to(new float[] {1}), null);
		}
	}
	private void prepareData() {
		outputs=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			outputs[i]=new float[weights[i].length];
		}
		
		layersNumber=weights.length;
		layersSize=new int[layersNumber+1];
		layersSize[0]=inputNumber;
		for(int i=1;i<=weights.length;i++) {
			layersSize[i]=weights[i-1].length;
		}
	}
	public void clearCPUData() {
		if(openCLLoaded) {
			weights=null;
			outputs=null;
		}else throw new NeuralException(0);
	}
	
	public final int getInputsNumber() {
		return inputNumber;
	}
	public int getOutputsNumber() {
		return outputs[outputs.length-1].length;
	}
	public final float[][][] getWeights() {
		return weights;
	}
	public final float[][] getOutputs() {
		if(openCLLoaded) {
			if(outputs==null) {
				float[][] outputs=new float[layersNumber][];
				for(int i=0;i<layersNumber;i++) {
					outputs[i]=new float[layersSize[i+1]];
					CL.clEnqueueReadBuffer(commandQueue, outputsCL[i], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*layersSize[i+1], Pointer.to(outputs[i]), 0, null, null);
				}
				return outputs;
			}
			for(int i=0;i<layersNumber;i++) {
				CL.clEnqueueReadBuffer(commandQueue, outputsCL[i], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*layersSize[i+1], Pointer.to(outputs[i]), 0, null, null);
			}
		}
		return outputs;
	}
	public final float[] getOutput() {
		if(openCLLoaded) {
			if(outputs==null) {
				float[] outputs=new float[layersSize[layersNumber]];
				CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(outputs), 0, null, null);
				return outputs;
			}
			CL.clEnqueueReadBuffer(commandQueue, outputsCL[layersNumber-1], CL.CL_TRUE, Sizeof.cl_float, Sizeof.cl_float*layersSize[layersNumber], Pointer.to(outputs[layersNumber-1]), 0, null, null);
		}
		return outputs[layersNumber-1];
	}
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	
	public void removeNeuron(int layer, int neuron) {
		outputs[layer]=new float[outputs[layer].length-1];
		layersSize[layer]--;
		float[][] temp=new float[outputs[layer].length][];
		System.arraycopy(weights[layer], 0, temp, 0, neuron);
		System.arraycopy(weights[layer], neuron+1, temp, neuron, temp.length-neuron);
		weights[layer]=temp;
		
		if(layer<weights.length-1) {
			layer++;
			float[] temp2;
			int max=weights[layer].length;
			for(int i=0;i<max;i++) {
				temp2=new float[weights[layer][i].length-1];
				System.arraycopy(weights[layer][i], 0, temp2, 0, neuron+1);
				System.arraycopy(weights[layer][i], neuron+2, temp2, neuron+1, temp2.length-neuron-1);
				weights[layer][i]=temp2;
			}
		}
	}
	
	public float[] simulate(float[] inputData) {
		if(inputData.length!=inputNumber)
			throw new NeuralException(4);
		
		if(openCLLoaded) {
			symFunction.applyOpenCL(inputData);
		}else {
			symFunction.apply(inputData);
		}
		return getOutput();
	}
	
	public abstract static class SymulationFunction{
		public abstract void apply(float[] inputData);
		public void applyOpenCL(float[] inputData) {
			throw new NeuralException(5);
		}
	}
	
	protected static final String openCLProgram=
			   "__kernel void simulate(__global const float *weights,__global const float *input,__global float *output,int connectionsNumber) {"
		      +"	int neuron=get_global_id(0);"
		      + "	int index=neuron*connectionsNumber;"
		      + "	neuron++;"								//offset of worksize in not supported by openCL yet(neuron[0]==bias)
		      + "	"
		      + "	float value=0;"
		      + "	for(int i=0;i<connectionsNumber;i++){"
		      + "		value+=weights[index+i]*input[i];"
		      + "	}"
		      + "	"
		      + "	output[neuron]=outputFunction(value);"
		      + "}";
}