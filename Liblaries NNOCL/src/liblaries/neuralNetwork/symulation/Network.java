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
import liblaries.neuralNetwork.functions.Function;

public class Network{
	private float[][][] weights;				//Input layer ID=0			[a][b][c] a->layer, b->neuron, c->connection
	private float[][] outputs;
	
	private int inputNumber;
	
	public Function function;
	
	int[] layersSize;
	int layersNumber;
	
	private boolean openCLLoaded=false;
	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_program program;
	private cl_kernel simulateKernel;
	private cl_kernel sumujKernel;
	
	private cl_mem[] weightsCL;							//[a] a->layer
	private cl_mem[] outputCL;							//[a]
	
	//public Network() {}			//to be or not to be?
	public Network(int inputNumber, float[][][] weights, Function function) {
		setWeights(inputNumber,weights);
		this.function=function;
	}
	public Network(int inputNumber, float[][][] weights, Function function, boolean loadOpenCL) {
		setWeights(inputNumber,weights);
		this.function=function;
		
		if(loadOpenCL)
			initializeOpenCL();
	}
	public void setWeights(int inputNumber,float[][][] weights) {
		this.inputNumber=inputNumber;
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
		
		program=CL.clCreateProgramWithSource(context, 2, new String[] {openCLprogram,function.getOpenCLProgram()}, null, null);
		CL.clBuildProgram(program, 1, new cl_device_id[] {devices[0]}, null, null, null);
		
		simulateKernel=CL.clCreateKernel(program, "sumOutput", null);
		sumujKernel=CL.clCreateKernel(program, "sumujWyjscia", null);
		
		createCLMem();
		
		openCLLoaded=true;
	}
	public void clearOpenCL() {
		openCLLoaded=false;
		
		for(int i=0;i<weights.length;i++) {
			CL.clReleaseMemObject(weightsCL[i]);
			CL.clReleaseMemObject(outputCL[i]);
		}
		
		CL.clReleaseKernel(sumujKernel);
		CL.clReleaseKernel(simulateKernel);
		CL.clReleaseProgram(program);
		CL.clReleaseCommandQueue(commandQueue);
		CL.clReleaseContext(context);
	}
	private void createCLMem() {
		weightsCL=new cl_mem[weights.length];
		outputCL=new cl_mem[weights.length];
		
		float[] wagiCLSrc;
		
		int index;
		for(int i=0;i<weights.length;i++) {
			int liczbaPo³¹czeñ=i==0?inputNumber:weights[i-1].length;
			wagiCLSrc=new float[weights[i].length*liczbaPo³¹czeñ];
			
			index=0;
			for(int j=0;j<weights[i].length;j++) {
				for(int k=0;k<weights[i][j].length;k++) {
					wagiCLSrc[index]=weights[i][j][k];
					index++;
				}
			}
			weightsCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*(index+1), Pointer.to(wagiCLSrc), null);
			
			outputCL[i]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_float*outputs[i].length, null, null);
		}
	}
	private void prepareData() {
		outputs=new float[weights.length][];
		
		for(int i=0;i<weights.length;i++) {
			outputs[i]=new float[weights[i].length];
		}
		
		layersSize=new int[weights.length];
		layersSize[0]=inputNumber;
		for(int i=1;i<=weights.length;i++) {
			layersSize[i]=weights[i].length;
		}
	}
	public void clearCPUData() {
		if(openCLLoaded)
			weights=null;
		else throw new NeuralException(0);
	}
	
	public final int getInputNumber() {
		return inputNumber;
	}
	public final float[][][] getWeight() {
		return weights;
	}
	public final float[][] getOutputs() {
		return outputs;
	}
	public final boolean isOpenCLLoaded() {
		return openCLLoaded;
	}
	
	
	public float[] symulujSieæ(float[] inputData) {		
		if(inputData.length!=inputNumber)
			throw new Error("Invalid input lenght. you use : "+inputData.length+" network lenght size: "+inputNumber);
		if(openCLLoaded) {
			cl_mem inputDataCL;
			for(int nrLayer=0;nrLayer<layersNumber;nrLayer++){
				if(nrLayer==0){																//Input layer
					inputDataCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*inputData.length, Pointer.to(inputData), null);
				}	
				else{
					inputDataCL=outputCL[nrLayer-1];
					inputData=outputs[nrLayer-1];
				}
				int neurony=layersSize[nrLayer+1];
				int connections=layersSize[nrLayer];
				
				cl_mem preWyjœciaCL=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_HOST_NO_ACCESS, Sizeof.cl_float*neurony*connections, null, null);
				Pointer preWyjœciaCLPtr=Pointer.to(preWyjœciaCL);
				
				CL.clSetKernelArg(simulateKernel, 0, Sizeof.cl_mem, Pointer.to(weightsCL[nrLayer]));
				CL.clSetKernelArg(simulateKernel, 1, Sizeof.cl_mem, Pointer.to(inputDataCL));
				CL.clSetKernelArg(simulateKernel, 2, Sizeof.cl_mem, preWyjœciaCLPtr);
				CL.clSetKernelArg(simulateKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, simulateKernel, 2, null ,new long[] {neurony,connections}, new long[]{1,1}, 0, null, null);
				
				CL.clSetKernelArg(sumujKernel, 0, Sizeof.cl_mem, preWyjœciaCLPtr);
				CL.clSetKernelArg(sumujKernel, 1, Sizeof.cl_mem, Pointer.to(outputCL[nrLayer]));
				CL.clSetKernelArg(sumujKernel, 2, Sizeof.cl_int, Pointer.to(new int[] {connections}));
				CL.clEnqueueNDRangeKernel(commandQueue, sumujKernel, 1, null, new long[] {neurony}, new long[] {1}, 0, null, null);
				
				CL.clEnqueueReadBuffer(commandQueue, outputCL[nrLayer], CL.CL_TRUE, 0, Sizeof.cl_float*neurony, Pointer.to(outputs[nrLayer]), 0, null, null);
				
				CL.clReleaseMemObject(inputDataCL);
				CL.clReleaseMemObject(preWyjœciaCL);
				
				for(int i=0;i<neurony;i++)
					outputs[nrLayer][i]=function.function(outputs[nrLayer][i]);
				
				CL.clReleaseMemObject(outputCL[nrLayer]);
				outputCL[nrLayer]=CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE|CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float*outputs[nrLayer].length, Pointer.to(outputs[nrLayer]), null);
				}
		}else {
			for(int nrLayer=0;nrLayer<weights.length;nrLayer++){
				if(nrLayer!=0){
					inputData=outputs[nrLayer-1];
				}
				
				for(int i=0;i<weights[nrLayer].length;i++) {
					outputs[nrLayer][i]=0;
					for(int j=0;j<weights[nrLayer][i].length;j++){
						outputs[nrLayer][i]+=weights[nrLayer][i][j]*inputData[j];
					}
					
					outputs[nrLayer][i]=function.function(outputs[nrLayer][i]);
				}
			}
		}
		return outputs[layersNumber-1];
	}
	
	private static final String openCLprogram=
		  "__kernel void symuluj(__global const float *wagi,__global const float *wejscia,__global float *preWyjscia,const int ilPolaczen) {"
		+ "		int polaczenie=get_global_id(1);"
		+ "		int index=get_global_id(0)*ilPolaczen+polaczenie;"
		+ "		"
		+ "		preWyjscia[index]=wejscia[polaczenie]*wagi[index];"
		+ "	}\n";
}