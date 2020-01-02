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

import java.util.List;
import java.util.function.IntSupplier;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.activationfunctions.DefaultDifferentiableActivationFunctionComponentImpl;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DefaultBatchNormDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DefaultDirectedAxonsComponentImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultComponentChainBatchImpl;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytoone.DefaultManyToOneFilterConcatDirectedComponentLegacy;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetomany.DefaultOneToManyDirectedComponentImpl;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.onetoone.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

/**
 * Mock implementation of DirectedComponentFactory
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentFactoryImpl implements DirectedComponentFactory {

	private MatrixFactory matrixFactory;
	private AxonsFactory axonsFactory;
	
	public DefaultDirectedComponentFactoryImpl(MatrixFactory matrixFactory, AxonsFactory axonsFactory) {
		this.matrixFactory = matrixFactory;
		this.axonsFactory = axonsFactory;
	}

	@Override
	public DirectedAxonsComponent<Neurons, Neurons, ?> createFullyConnectedAxonsComponent(Neurons leftNeurons,
			Neurons rightNeurons, Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createFullyConnectedAxons(leftNeurons, rightNeurons, connectionWeights, biases));
	}

	@Override
	public <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R, ?> createDirectedAxonsComponent(
			Axons<L, R, ?> axons) {
		return new DefaultDirectedAxonsComponentImpl<>(axons);
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createConvolutionalAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config,
			Matrix connectionWeights, Matrix biases) {
		return createDirectedAxonsComponent(axonsFactory.createConvolutionalAxons(leftNeurons, rightNeurons, config, connectionWeights, biases));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createMaxPoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config,
			boolean scaleOutputs) {
		return createDirectedAxonsComponent(axonsFactory.createMaxPoolingAxons(leftNeurons, rightNeurons, scaleOutputs, config));
	}

	@Override
	public DirectedAxonsComponent<Neurons3D, Neurons3D, ?> createAveragePoolingAxonsComponent(Neurons3D leftNeurons,
			Neurons3D rightNeurons, Axons3DConfig config) {
		return createDirectedAxonsComponent(axonsFactory.createAveragePoolingAxons(leftNeurons, rightNeurons, 
				config));
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons) {
		throw new UnsupportedOperationException();
	}

	@Override
	public <N extends Neurons> BatchNormDirectedAxonsComponent<N, ?> createBatchNormAxonsComponent(N leftNeurons,
			N rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		throw new UnsupportedOperationException();
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(axonsFactory.createScaleAndShiftAxons(leftNeurons, 
				rightNeurons, null, null), null, null);
	}

	@Override
	public BatchNormDirectedAxonsComponent<Neurons3D, ?> createConvolutionalBatchNormAxonsComponent(
			Neurons3D leftNeurons, Neurons3D rightNeurons, Matrix gamma, Matrix beta, Matrix mean, Matrix stddev) {
		return new DefaultBatchNormDirectedAxonsComponentImpl<>(axonsFactory.createScaleAndShiftAxons(leftNeurons, rightNeurons, 
				expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, gamma), 
				expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, beta)), expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, mean), 
				expandChannelValuesToFeatureValues(matrixFactory, rightNeurons, stddev));
	}
	
	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons3D rightNeurons, Matrix channelValues) {
		if (channelValues == null) return null;
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
	public <N extends Neurons> DirectedAxonsComponent<N, N, ?> createPassThroughAxonsComponent(N leftNeurons,
			N rightNeurons) {
		return createDirectedAxonsComponent(new PassThroughAxonsImpl<>(leftNeurons, rightNeurons));
	}

	@Override
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(IntSupplier targetComponentsCount) {
		return new DefaultOneToManyDirectedComponentImpl(targetComponentsCount);
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			PathCombinationStrategy pathCombinationStrategy) {
		return new DefaultManyToOneFilterConcatDirectedComponentLegacy(pathCombinationStrategy);
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons, 
			DifferentiableActivationFunction differentiableActivationFunction) {
		return new DefaultDifferentiableActivationFunctionComponentImpl(neurons, differentiableActivationFunction);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(Neurons leftNeurons, Neurons rightNeurons,
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		return new DefaultDirectedComponentChainBipoleGraphImpl(this, leftNeurons, rightNeurons, parallelComponentChainsBatch, pathCombinationStrategy);
	}

	@Override
	public DefaultDirectedComponentChain createDirectedComponentChain(
			List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		return new DefaultDirectedComponentChainImpl(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentChainBatch createDirectedComponentChainBatch(
			List<DefaultDirectedComponentChain> parallelComponents) {
		return new DefaultComponentChainBatchImpl(parallelComponents);
	}
}
