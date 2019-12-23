package org.ml4j.nn.factories;

import java.util.List;
import java.util.function.IntSupplier;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.activationfunctions.DefaultDifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DefaultBatchNormDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DefaultDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.legacy.ConvolutionalBatchNormDirectedAxonsComponentImpl2;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchImpl2;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.manytoone.legacy.DefaultManyToOneFilterConcatDirectedComponentLegacy;
import org.ml4j.nn.components.onetomany.DefaultOneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainBipoleGraphImpl2;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of DirectedComponentFactory
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentFactoryImpl2 implements DirectedComponentFactory {

	
	private AxonsFactory axonsFactory;
	private MatrixFactory matrixFactory;
	
	/**
	 * @param matrixFactory The matrix factory.
	 * @param axonsFactory The axons factory.
	 */
	public DefaultDirectedComponentFactoryImpl2(MatrixFactory matrixFactory, AxonsFactory axonsFactory) {
		this.axonsFactory = axonsFactory;
		this.matrixFactory = matrixFactory;
	}
	
	@Override
	public DirectedAxonsComponent<Neurons, Neurons> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createFullyConnectedAxons(leftNeurons, rightNeurons, connectionWeights, biases));
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponent(
			Axons<L, R, ?> axons) {
		return new DefaultDirectedAxonsComponentImpl<>(axons);
	}
	
	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createConvolutionalAxons(leftNeurons, rightNeurons,
				strideWidth, strideHeight, paddingWidth, paddingHeight, connectionWeights, biases));
	}
	
	

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			boolean scaleOutputs) {
		return createDirectedAxonsComponent(axonsFactory.createMaxPoolingAxons(leftNeurons, rightNeurons, scaleOutputs, strideWidth, strideHeight, paddingWidth, paddingHeight));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight) {
		return createDirectedAxonsComponent(axonsFactory.createAveragePoolingAxons(leftNeurons, rightNeurons, strideWidth, strideHeight, paddingWidth, paddingHeight));
	}
	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, null, null));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons, Matrix gamma, Matrix beta, Matrix means, Matrix stdev) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, gamma, beta));
	}
	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		return new ConvolutionalBatchNormDirectedAxonsComponentImpl2(
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, null, null));
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons, Matrix gamma, Matrix beta, Matrix means, Matrix stdev) {
		Matrix expandedGamma = gamma == null ? null
				: expandChannelValuesToFeatureValues(matrixFactory,
						rightNeurons, gamma);
		Matrix expandedBeta = beta == null ? null
				: expandChannelValuesToFeatureValues(matrixFactory,
						rightNeurons, beta);
		return new ConvolutionalBatchNormDirectedAxonsComponentImpl2(matrixFactory,
				axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, expandedGamma, expandedBeta), means,
				stdev);
	}
	
	
	public  Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons3D rightNeurons, Matrix channelValues) {
		float[] channelValuesArray = channelValues.getRowByRowArray();
		float[] channelValuesExpanded = new float[rightNeurons.getNeuronCountExcludingBias()];
		int index = 0;
		for (int channel = 0; channel < channelValuesArray.length; channel++) {
			for (int i = 0;  i < rightNeurons.getWidth() * rightNeurons.getHeight(); i++) {
				channelValuesExpanded[index++] = channelValuesArray[channel];
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(channelValuesExpanded.length, 1, channelValuesExpanded);

	}

	@Override
	public <N extends Neurons> DirectedAxonsComponent<N, N> createPassThroughAxonsComponent(N leftNeurons,
			N rightNeurons) {
		throw new UnsupportedOperationException();
		//return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(IntSupplier targetComponentsCount) {
		return new DefaultOneToManyDirectedComponent(targetComponentsCount);
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			PathCombinationStrategy pathCombinationStrategy) {
		if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT) {
			return new DefaultManyToOneFilterConcatDirectedComponentLegacy(pathCombinationStrategy);
		} else {
			throw new UnsupportedOperationException();
			//return new DefaultManyToOneAdditionDirectedComponent();
		}
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons, 
			DifferentiableActivationFunction differentiableActivationFunction) {
		return new DefaultDifferentiableActivationFunctionDirectedComponentImpl(neurons, differentiableActivationFunction);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(Neurons leftNeurons, Neurons rightNeurons,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		return new DefaultDirectedComponentChainBipoleGraphImpl2(this, leftNeurons, rightNeurons, parallelComponentChainsBatch, pathCombinationStrategy);
	}

	@Override
	public DefaultDirectedComponentChain createDirectedComponentChain(
			List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		return new DefaultDirectedComponentChainImpl(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> createDirectedComponentChainBatch(
			List<DefaultDirectedComponentChain> parallelComponents) {
		return new DefaultDirectedComponentChainBatchImpl2(parallelComponents);
	}

}
