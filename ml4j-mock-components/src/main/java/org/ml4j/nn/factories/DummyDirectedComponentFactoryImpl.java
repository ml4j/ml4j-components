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
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.NoOpAxons;
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

/**
 * Mock implementation of DirectedComponentFactory
 * 
 * @author Michael Lavelle
 */
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

	private <L extends Neurons, R extends Neurons> DirectedAxonsComponent<L, R> createDirectedAxonsComponent(
			L leftNeurons, R rightNeurons) {
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
	public OneToManyDirectedComponent<?> createOneToManyDirectedComponent(IntSupplier targetComponentsCount) {
		return new DummyOneToManyDirectedComponent(targetComponentsCount);
	}

	@Override
	public ManyToOneDirectedComponent<?> createManyToOneDirectedComponent(
			PathCombinationStrategy pathCombinationStrategy) {
		return new DummyManyToOneDirectedComponent(pathCombinationStrategy);
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons, 
			DifferentiableActivationFunction differentiableActivationFunction) {
		return new DummyDifferentiableActivationFunctionComponent(neurons, differentiableActivationFunction);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph createDirectedComponentBipoleGraph(Neurons leftNeurons, Neurons rightNeurons,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		return new DummyDefaultDirectedComponentChainBipoleGraph(leftNeurons, rightNeurons, parallelComponentChainsBatch);
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
		return new NoOpAxons<>(leftNeurons, rightNeurons);
	}

}
