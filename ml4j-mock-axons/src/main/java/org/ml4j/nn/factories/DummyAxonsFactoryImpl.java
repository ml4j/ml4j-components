package org.ml4j.nn.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.BiasMatrix;
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
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			WeightsMatrix connectionWeights, BiasMatrix biases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(Neurons leftNeurons, Neurons rightNeurons,
			WeightsMatrix connectionWeights, BiasMatrix leftToRightBiases, BiasMatrix rightToLeftBiases) {
		return new DummyFullyConnectedAxonsImpl(matrixFactory, leftNeurons, rightNeurons);
	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		return new DummyAveragePoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config, WeightsMatrix connectionWeights, BiasMatrix biases) {
		return new DummyConvolutionalAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(Neurons3D leftNeurons, Neurons3D rightNeurons, boolean scaleOutputs,
			Axons3DConfig config) {
		return new DummyMaxPoolingAxonsImpl(matrixFactory, leftNeurons, rightNeurons, config);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(N leftNeurons, N rightNeurons,
			WeightsMatrix gamma, BiasMatrix beta) {
		return new DummyScaleAndShiftAxonsImpl<>(matrixFactory, leftNeurons, rightNeurons);
	}

}
