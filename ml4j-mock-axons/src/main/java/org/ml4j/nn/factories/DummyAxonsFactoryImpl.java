package org.ml4j.nn.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.axons.mocks.DummyAveragePoolingAxonsImpl;
import org.ml4j.nn.axons.mocks.DummyConvolutionalAxonsImpl;
import org.ml4j.nn.axons.mocks.DummyFullyConnectedAxonsImpl;
import org.ml4j.nn.axons.mocks.DummyMaxPoolingAxonsImpl;
import org.ml4j.nn.axons.mocks.DummyScaleAndShiftAxonsImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;


public class DummyAxonsFactoryImpl implements AxonsFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;

	public DummyAxonsFactoryImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(AxonsConfig<Neurons, Neurons> axonsConfig,
			WeightsMatrix connectionWeights, BiasVector biases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(AxonsConfig<Neurons, Neurons> axonsConfig,
			WeightsMatrix connectionWeights, BiasVector leftToRightBiases, BiasVector rightToLeftBiases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());
	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(
			Axons3DConfig config) {
		return new DummyAveragePoolingAxonsImpl(matrixFactory, config.getLeftNeurons(), config.getRightNeurons(), config);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(
			Axons3DConfig config, WeightsMatrix connectionWeights, BiasVector biases) {
		return new DummyConvolutionalAxonsImpl(matrixFactory, config.getLeftNeurons(), config.getRightNeurons(), config);
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(
			Axons3DConfig config, boolean scaleOutputs) {
		return new DummyMaxPoolingAxonsImpl(matrixFactory, config.getLeftNeurons(), config.getRightNeurons(), config);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(AxonsConfig<N, N> axonsConfig, 
			WeightsMatrix gamma, BiasVector beta) {
		return new DummyScaleAndShiftAxonsImpl<>(matrixFactory, axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>> A createAxons(AxonsType axonsType, Class<A> axonsClass,
			AxonsConfig<Neurons, Neurons> axonsConfig) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>> A createAxons3D(AxonsType axonsType, Class<A> axonsClass,
			AxonsConfig<Neurons3D, Neurons3D> axonsConfig) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>> A createAxons3DWith3DConfig(AxonsType axonsType,
			Class<A> axonsClass, Axons3DConfig axonsConfig) {
		throw new UnsupportedOperationException();
	}

	

}
