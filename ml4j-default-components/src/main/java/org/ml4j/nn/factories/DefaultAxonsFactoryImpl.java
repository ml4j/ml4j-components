/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.factories;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AveragePoolingAxons;
import org.ml4j.nn.axons.AxonWeightsInitialiser;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsBaseType;
import org.ml4j.nn.axons.AxonsConfig;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.axons.DefaultAveragePoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultFullyConnectedAxonWeightsInitialiser;
import org.ml4j.nn.axons.DefaultFullyConnectedAxonsImpl;
import org.ml4j.nn.axons.DefaultMaxPoolingAxonsImpl;
import org.ml4j.nn.axons.DefaultOneByOneConvolutionalAxonsImpl;
import org.ml4j.nn.axons.DefaultScaleAndShiftAxonWeightsInitialiser;
import org.ml4j.nn.axons.DefaultScaleAndShiftAxonsImpl;
import org.ml4j.nn.axons.DefaultSpaceToDepthAxons;
import org.ml4j.nn.axons.FeaturesVectorFormat;
import org.ml4j.nn.axons.FullyConnectedAxonWeightsImpl;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.MaxPoolingAxons;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.axons.ScaleAndShiftAxonWeightsImpl;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Default implementation of AxonsFactory, used to generate Axons
 * implementations.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultAxonsFactoryImpl implements AxonsFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	protected MatrixFactory matrixFactory;

	/**
	 * @param matrixFactory The MatrixFactory used to construct matrices used by the
	 *                      generated Axons implementations.
	 */
	public DefaultAxonsFactoryImpl(MatrixFactory matrixFactory) {
		this.matrixFactory = matrixFactory;
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(AxonsConfig<Neurons, Neurons> axonsConfig,
			WeightsMatrix connectionWeights, BiasVector biases) {
		return createFullyConnectedAxons(axonsConfig, connectionWeights, biases, null);
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(AxonsConfig<Neurons, Neurons> axonsConfig,
			WeightsMatrix connectionWeights, BiasVector leftToRightBiases, BiasVector rightToLeftBiases) {

		if (connectionWeights == null || connectionWeights.getFormat() == null) {
			throw new IllegalArgumentException("Connection weights format cannot be null");
		}

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultFullyConnectedAxonWeightsInitialiser(
				axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());

		WeightsMatrix initialConnectionWeights = connectionWeights.getMatrix() == null ? new WeightsMatrixImpl(
				axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory), connectionWeights.getFormat())
				: connectionWeights;
		Optional<Matrix> initialLeftToRightBiasMatrix = leftToRightBiases == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(leftToRightBiases.getVector());
		Optional<Matrix> initialRightToLeftBiasMatrix = rightToLeftBiases == null
				? axonWeightsInitialiser.getInitialRightToLeftBiases(matrixFactory)
				: Optional.of(rightToLeftBiases.getVector());

		Optional<BiasVector> initialLeftToRightBiases = initialLeftToRightBiasMatrix.isPresent()
				? Optional.of(new BiasVectorImpl(initialLeftToRightBiasMatrix.get(), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT))
				: Optional.empty();

		Optional<BiasVector> initialRightToLeftBiases = initialRightToLeftBiasMatrix.isPresent()
				? Optional.of(new BiasVectorImpl(initialRightToLeftBiasMatrix.get(), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT))
				: Optional.empty();

		return new DefaultFullyConnectedAxonsImpl(axonsConfig,
				new FullyConnectedAxonWeightsImpl(axonsConfig.getLeftNeurons().getNeuronCountExcludingBias(),
						axonsConfig.getRightNeurons().getNeuronCountExcludingBias(), initialConnectionWeights,
						axonsConfig.getLeftNeurons().hasBiasUnit() && initialLeftToRightBiases.isPresent()
								? initialLeftToRightBiases.get()
								: null,
						axonsConfig.getRightNeurons().hasBiasUnit() && initialRightToLeftBiases.isPresent()
								? initialRightToLeftBiases.get()
								: null));
	}

	@Override
	public AveragePoolingAxons createAveragePoolingAxons(Axons3DConfig config) {
		return new DefaultAveragePoolingAxonsImpl(matrixFactory, config);
	}

	@Override
	public ConvolutionalAxons createConvolutionalAxons(Axons3DConfig config, WeightsMatrix connectionWeights,
			BiasVector biases) {
		if (DefaultOneByOneConvolutionalAxonsImpl.isEligible(config)) {

			return new DefaultOneByOneConvolutionalAxonsImpl(this, config, connectionWeights, biases);
		} else {
			return new DefaultConvolutionalAxonsImpl(this, config, connectionWeights, biases);
		}
	}

	@Override
	public MaxPoolingAxons createMaxPoolingAxons(Axons3DConfig config, boolean scaleOutputs) {
		return new DefaultMaxPoolingAxonsImpl(matrixFactory, config, scaleOutputs);
	}

	@Override
	public <N extends Neurons> ScaleAndShiftAxons<N> createScaleAndShiftAxons(AxonsConfig<N, N> axonsConfig,
			WeightsMatrix gamma, BiasVector beta) {

		if (gamma == null) {
			throw new IllegalArgumentException("Gamma cannot be null");
		}

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultScaleAndShiftAxonWeightsInitialiser(
				axonsConfig.getLeftNeurons());

		Matrix initialGammaMatrix = gamma.getMatrix() == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: gamma.getMatrix();
		Optional<Matrix> initialBeta = beta == null ? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(beta.getVector());

		WeightsMatrix intialGamma = new WeightsMatrixImpl(initialGammaMatrix, gamma.getFormat());

		ScaleAndShiftAxonWeightsImpl weights = new ScaleAndShiftAxonWeightsImpl(
				axonsConfig.getLeftNeurons().getNeuronCountExcludingBias(),
				axonsConfig.getRightNeurons().getNeuronCountExcludingBias(), intialGamma,
				initialBeta.isPresent() ? new BiasVectorImpl(initialBeta.get(), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT) : null, null);
		return new DefaultScaleAndShiftAxonsImpl<>(axonsConfig, weights);
	}

	@Override
	public <A extends Axons<Neurons, Neurons, ?>> A createAxons(AxonsType axonsType, Class<A> axonsClass, AxonsConfig<Neurons, Neurons> axonsConfig) {
		if (axonsType.getBaseType().equals(AxonsBaseType.FULLY_CONNECTED) && axonsClass.isAssignableFrom(FullyConnectedAxons.class)) {
			return axonsClass.cast(createFullyConnectedAxons(axonsConfig, null, null));
		} else if (axonsType.getBaseType().equals(AxonsBaseType.SCALE_AND_SHIFT) && axonsClass.isAssignableFrom(ScaleAndShiftAxons.class)) {
			return axonsClass.cast(createScaleAndShiftAxons(axonsConfig, null, null));
		} else if (axonsType.getBaseType().equals(AxonsBaseType.PASS_THROUGH) && axonsClass.isAssignableFrom(PassThroughAxonsImpl.class) ) {
			return axonsClass.cast(new PassThroughAxonsImpl<>(axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons()));
		} else {
			throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
		}
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>> A createAxons3D(AxonsType axonsType, Class<A> axonsClass, AxonsConfig<Neurons3D, Neurons3D> config) {
		if (axonsType.equals(DefaultSpaceToDepthAxons.SPACE_TO_DEPTH_AXONS_TYPE) && axonsClass.isAssignableFrom(DefaultSpaceToDepthAxons.class)) {
			return axonsClass.cast(new DefaultSpaceToDepthAxons(config));
		}
		throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
	}

	@Override
	public <A extends Axons<Neurons3D, Neurons3D, ?>> A createAxons3DWith3DConfig(AxonsType axonsType, Class<A> axonsClass, Axons3DConfig axonsConfig) {
		if (axonsType.getBaseType().equals(AxonsBaseType.CONVOLUTIONAL) && axonsClass.isAssignableFrom(ConvolutionalAxons.class)) {
			return axonsClass.cast(createConvolutionalAxons(axonsConfig, null, null));
		} else if (axonsType.getBaseType().equals(AxonsBaseType.MAX_POOLING) && axonsClass.isAssignableFrom(MaxPoolingAxons.class)) {
			return axonsClass.cast(createMaxPoolingAxons(axonsConfig, false));
		} else if (axonsType.getBaseType().equals(AxonsBaseType.AVERAGE_POOLING) && axonsClass.isAssignableFrom(AveragePoolingAxons.class)) {
			return axonsClass.cast(createAveragePoolingAxons(axonsConfig));
		} else if (axonsType.getBaseType().equals(AxonsBaseType.PASS_THROUGH) && axonsClass.isAssignableFrom(PassThroughAxonsImpl.class)) {
			return axonsClass.cast(new PassThroughAxonsImpl<>(axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons()));
		} else {
			throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
		}
	}
}
