package org.ml4j.nn.factories;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.DummyAxons;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DummyDifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DummyBatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DummyDirectedAxonsComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DummyDefaultComponentChainBatch;
import org.ml4j.nn.components.manytoone.DummyManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetomany.DummyOneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.DummyDefaultDirectedComponentChain;
import org.ml4j.nn.components.onetoone.DummyDefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedComponentFactoryImpl implements DirectedComponentFactory {

	@Override
	public DirectedAxonsComponent<Neurons, Neurons> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponent(
			Axons<L, R, ?> axons) {
		return createDirectedAxonsComponent(axons.getLeftNeurons(), axons.getRightNeurons());
	}
	
	private <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponent(L leftNeurons, R rightNeurons) {
		return new DummyDirectedAxonsComponent<>(createDummyAxons(leftNeurons, rightNeurons));
	}
	
	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight,
			boolean scaleOutputs) {
		return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, int strideWidth, int strideHeight, Integer paddingWidth, Integer paddingHeight) {
		return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, N> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		throw new UnsupportedOperationException();
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		return new DummyBatchNormDirectedAxonsComponent<>(createDummyAxons(leftNeurons, rightNeurons));
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, Neurons3D> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		return new DummyBatchNormDirectedAxonsComponent<>(createDummyAxons(leftNeurons, rightNeurons));
	}

	@Override
	public <N extends Neurons> DirectedAxonsComponent<N, N> createPassThroughAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return createDirectedAxonsComponent(leftNeurons, rightNeurons);
	}

	@Override
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(
			List<? extends ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		return new DummyOneToManyDirectedComponent();
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			PathCombinationStrategy pathCombinationStrategy) {
		return new DummyManyToOneDirectedComponent();
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(
			DifferentiableActivationFunction differentiableActivationFunction) {
			return new DummyDifferentiableActivationFunctionComponent(differentiableActivationFunction);
	}


	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		return new DummyDefaultDirectedComponentChainBipoleGraph(parallelComponentChainsBatch);
	}


	@Override
	public DefaultDirectedComponentChain createDirectedComponentChain(
			List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		return new DummyDefaultDirectedComponentChain(sequentialComponents);
	}


	@Override
	public DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> createDirectedComponentChainBatch(
			List<DefaultDirectedComponentChain> parallelComponents) {
		return new DummyDefaultComponentChainBatch(parallelComponents);
	}
	
	private <L extends Neurons, R extends Neurons> Axons<L, R, ?> createDummyAxons(L leftNeurons, R rightNeurons) {
		return new DummyAxons<>(leftNeurons, rightNeurons);
	}
	
	
}
