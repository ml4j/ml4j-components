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
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.BiasMatrixImpl;
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
			WeightsMatrix connectionWeights, BiasMatrix biases) {
		return createFullyConnectedAxons(axonsConfig, connectionWeights, biases, null);
	}

	@Override
	public FullyConnectedAxons createFullyConnectedAxons(AxonsConfig<Neurons, Neurons> axonsConfig,
			WeightsMatrix connectionWeights, BiasMatrix leftToRightBiases, BiasMatrix rightToLeftBiases) {

		if (connectionWeights == null || connectionWeights.getFormat() == null) {
			throw new IllegalArgumentException("Connection weights format cannot be null");
		}

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultFullyConnectedAxonWeightsInitialiser(
				axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());

		WeightsMatrix initialConnectionWeights = connectionWeights.getWeights() == null ? new WeightsMatrixImpl(
				axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory), connectionWeights.getFormat())
				: connectionWeights;
		Optional<Matrix> initialLeftToRightBiasMatrix = leftToRightBiases == null
				? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(leftToRightBiases.getWeights());
		Optional<Matrix> initialRightToLeftBiasMatrix = rightToLeftBiases == null
				? axonWeightsInitialiser.getInitialRightToLeftBiases(matrixFactory)
				: Optional.of(rightToLeftBiases.getWeights());

		Optional<BiasMatrix> initialLeftToRightBiases = initialLeftToRightBiasMatrix.isPresent()
				? Optional.of(new BiasMatrixImpl(initialLeftToRightBiasMatrix.get()))
				: Optional.empty();

		Optional<BiasMatrix> initialRightToLeftBiases = initialRightToLeftBiasMatrix.isPresent()
				? Optional.of(new BiasMatrixImpl(initialRightToLeftBiasMatrix.get()))
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
			BiasMatrix biases) {
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
			WeightsMatrix gamma, BiasMatrix beta) {

		if (gamma == null) {
			throw new IllegalArgumentException("Gamma cannot be null");
		}

		AxonWeightsInitialiser axonWeightsInitialiser = new DefaultScaleAndShiftAxonWeightsInitialiser(
				axonsConfig.getLeftNeurons());

		Matrix initialGammaMatrix = gamma.getWeights() == null
				? axonWeightsInitialiser.getInitialConnectionWeights(matrixFactory)
				: gamma.getWeights();
		Optional<Matrix> initialBeta = beta == null ? axonWeightsInitialiser.getInitialLeftToRightBiases(matrixFactory)
				: Optional.of(beta.getWeights());

		WeightsMatrix intialGamma = new WeightsMatrixImpl(initialGammaMatrix, gamma.getFormat());

		ScaleAndShiftAxonWeightsImpl weights = new ScaleAndShiftAxonWeightsImpl(
				axonsConfig.getLeftNeurons().getNeuronCountExcludingBias(),
				axonsConfig.getRightNeurons().getNeuronCountExcludingBias(), intialGamma,
				initialBeta.isPresent() ? new BiasMatrixImpl(initialBeta.get()) : null, null);
		return new DefaultScaleAndShiftAxonsImpl<>(axonsConfig, weights);
	}

	@Override
	public Axons<Neurons, Neurons, ?> createAxons(AxonsType axonsType, AxonsConfig<Neurons, Neurons> axonsConfig) {
		if (axonsType.getBaseType().equals(AxonsBaseType.FULLY_CONNECTED)) {
			return createFullyConnectedAxons(axonsConfig, null, null);
		} else if (axonsType.getBaseType().equals(AxonsBaseType.SCALE_AND_SHIFT)) {
			return createScaleAndShiftAxons(axonsConfig, null, null);
		} else if (axonsType.getBaseType().equals(AxonsBaseType.PASS_THROUGH)) {
			return new PassThroughAxonsImpl<>(axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());
		} else {
			throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
		}
	}

	@Override
	public Axons<Neurons3D, Neurons3D, ?> createAxons3D(AxonsType axonsType, AxonsConfig<Neurons3D, Neurons3D> config) {
		if (axonsType.equals(DefaultSpaceToDepthAxons.SPACE_TO_DEPTH_AXONS_TYPE)) {
			return new DefaultSpaceToDepthAxons(config);
		}
		throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
	}

	@Override
	public Axons<Neurons3D, Neurons3D, ?> createAxons3DWith3DConfig(AxonsType axonsType, Axons3DConfig axonsConfig) {
		if (axonsType.getBaseType().equals(AxonsBaseType.CONVOLUTIONAL)) {
			return createConvolutionalAxons(axonsConfig, null, null);
		} else if (axonsType.getBaseType().equals(AxonsBaseType.MAX_POOLING)) {
			return createMaxPoolingAxons(axonsConfig, false);
		} else if (axonsType.getBaseType().equals(AxonsBaseType.AVERAGE_POOLING)) {
			return createAveragePoolingAxons(axonsConfig);
		} else if (axonsType.getBaseType().equals(AxonsBaseType.PASS_THROUGH)) {
			return new PassThroughAxonsImpl<>(axonsConfig.getLeftNeurons(), axonsConfig.getRightNeurons());
		} else {
			throw new IllegalArgumentException("Unable to create axons of type:" + axonsType);
		}
	}
}
