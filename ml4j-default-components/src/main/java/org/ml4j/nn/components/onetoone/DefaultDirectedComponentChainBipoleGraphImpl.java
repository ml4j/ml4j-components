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
package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DefaultDirectedComponentBipoleGraph, consisting of
 * a parallel edges ( DefaultDirectedComponentChainBatch ), with the paths
 * through those parallel edges all starting at the same point in the network
 * and ending at the same point in the network.
 * 
 * The activations at the input of this graph are mapped to the
 * DefaultDirectedComponentChainBatch activations via a
 * OneToManyDirectedComponent, and the DefaultDirectedComponentChainBatch output
 * activations are mapped to the output of this graph via a
 * ManyToOneDirectedComponent, using a specified PathCombinationStrategy.
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentChainBipoleGraphImpl extends DefaultDirectedComponentChainBipoleGraphBase
		implements DefaultDirectedComponentChainBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBipoleGraphImpl.class);

	private OneToManyDirectedComponent<?> oneToManyDirectedComponent;
	private ManyToOneDirectedComponent<?> manyToOneDirectedComponent;

	/**
	 * 
	 * @param directedComponentFactory     A DirectedComponentFactory instance, used
	 *                                     to construct the nested
	 *                                     OneToManyDirectedComponent and
	 *                                     ManyToOneDirectedComponent.
	 * @param inputNeurons                 The input neurons of this graph.
	 * @param outputNeurons                The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this
	 *                                     graph, connecting
	 * @param pathCombinationStrategy      The strategy specifying how the outputs
	 *                                     of the parallel edges should be combined
	 *                                     to produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(String name, DirectedComponentFactory directedComponentFactory,
			Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		this(name, inputNeurons, outputNeurons, parallelComponentChainsBatch,
				parallelComponentChainsBatch.getComponents().size() == 1 ? null
						: directedComponentFactory.createOneToManyDirectedComponent(
								() -> parallelComponentChainsBatch.getComponents().size()),
				parallelComponentChainsBatch.getComponents().size() == 1 ? null
						: directedComponentFactory.createManyToOneDirectedComponent(outputNeurons,
								pathCombinationStrategy),
				pathCombinationStrategy);
	}

	/**
	 * 
	 * @param directedComponentFactory     A DirectedComponentFactory instance, used
	 *                                     to construct the nested
	 *                                     OneToManyDirectedComponent and
	 *                                     ManyToOneDirectedComponent.
	 * @param inputNeurons                 The input neurons of this graph.
	 * @param outputNeurons                The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this
	 *                                     graph, connecting
	 * @param pathCombinationStrategy      The strategy specifying how the outputs
	 *                                     of the parallel edges should be combined
	 *                                     to produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(String name, DirectedComponentFactory directedComponentFactory,
			Neurons inputNeurons, Neurons3D outputNeurons,
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch,
			PathCombinationStrategy pathCombinationStrategy) {
		this(name, inputNeurons, outputNeurons, parallelComponentChainsBatch,
				parallelComponentChainsBatch.getComponents().size() == 1 ? null
						: directedComponentFactory.createOneToManyDirectedComponent(
								() -> parallelComponentChainsBatch.getComponents().size()),
				parallelComponentChainsBatch.getComponents().size() == 1 ? null
						: directedComponentFactory.createManyToOneDirectedComponent3D(outputNeurons,
								pathCombinationStrategy),
				pathCombinationStrategy);
	}

	/**
	 * 
	 * @param name						   The component name.
	 * @param inputNeurons                 The input neurons of this graph.
	 * @param outputNeurons                The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this
	 *                                     graph, connecting
	 * @param pathCombinationStrategy      The strategy specifying how the outputs
	 *                                     of the parallel edges should be combined
	 *                                     to produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(String name, Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch,
			OneToManyDirectedComponent<?> oneToManyDirectedComponent,
			ManyToOneDirectedComponent<?> manyToOneDirectedComponent, PathCombinationStrategy pathCombinationStrategy) {
		super(name, inputNeurons, outputNeurons, parallelComponentChainsBatch, pathCombinationStrategy);
		this.oneToManyDirectedComponent = oneToManyDirectedComponent;
		this.manyToOneDirectedComponent = manyToOneDirectedComponent;
	}

	@Override
	public DefaultDirectedComponentChainBatch getEdges() {
		return parallelComponentChainsBatch;
	}

	@Override
	public DefaultDirectedComponentChainBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {

		boolean originalInputIsImmutable = neuronsActivation.isImmutable();

		if (neuronsActivation.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException(
					neuronsActivation.getFeatureCount() + ":" + getInputNeurons().getNeuronCountExcludingBias());
		}
		LOGGER.debug("Forward propagating through DefaultDirectedComponentChainBipoleGraphImpl");

		if (parallelComponentChainsBatch.getComponents().size() == 1) {
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch
					.forwardPropagate(Arrays.asList(neuronsActivation), context);
			return new DefaultDirectedComponentChainBipoleGraphActivationImpl(this, parallelChainsActivation,
					parallelChainsActivation.getOutput().get(0), originalInputIsImmutable);

		} else {

			OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = oneToManyDirectedComponent
					.forwardPropagate(neuronsActivation, context);

			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch
					.forwardPropagate(oneToManyDirectedComponentActivation.getOutput(), context);

			ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = manyToOneDirectedComponent
					.forwardPropagate(parallelChainsActivation.getOutput(), context);
			if (manyToOneDirectedComponentActivation.getOutput().getFeatureCount() != getOutputNeurons()
					.getNeuronCountExcludingBias()) {
				throw new IllegalArgumentException("Many to one activation feature count of:"
						+ manyToOneDirectedComponentActivation.getOutput().getFeatureCount()
						+ " does not match output neuron count of:" + getOutputNeurons().getNeuronCountExcludingBias());
			}
			return new DefaultDirectedComponentChainBipoleGraphActivationImpl(this,
					oneToManyDirectedComponentActivation, parallelChainsActivation,
					manyToOneDirectedComponentActivation, originalInputIsImmutable);

		}
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}
	
	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		allComponentsIncludingThis.addAll(this.parallelComponentChainsBatch.getComponents().stream().flatMap(c -> c.flatten().stream()).collect(Collectors.toSet()));
		return allComponentsIncludingThis;
	}

	@Override
	public DefaultDirectedComponentChainBipoleGraph dup(DirectedComponentFactory directedComponentFactory) {
		return new DefaultDirectedComponentChainBipoleGraphImpl(name, inputNeurons, outputNeurons,
				parallelComponentChainsBatch.dup(directedComponentFactory), oneToManyDirectedComponent.dup(directedComponentFactory), manyToOneDirectedComponent.dup(directedComponentFactory),
				pathCombinationStrategy);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		if (oneToManyDirectedComponent != null && manyToOneDirectedComponent != null) {
			return NeuronsActivationFormat
					.intersectOptionals(Arrays.asList(oneToManyDirectedComponent.optimisedFor(),
							parallelComponentChainsBatch.optimisedFor(), manyToOneDirectedComponent.optimisedFor()));
		} else if (oneToManyDirectedComponent != null) {
			return NeuronsActivationFormat
					.intersectOptionals(Arrays.asList(oneToManyDirectedComponent.optimisedFor(),
							parallelComponentChainsBatch.optimisedFor()));
		} else if (manyToOneDirectedComponent != null) {
			return NeuronsActivationFormat
					.intersectOptionals(Arrays.asList(manyToOneDirectedComponent.optimisedFor(),
							parallelComponentChainsBatch.optimisedFor()));
		} else {
			return parallelComponentChainsBatch.optimisedFor();
		}
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return (oneToManyDirectedComponent == null || oneToManyDirectedComponent.isSupported(format))
				&& NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation())
				&& parallelComponentChainsBatch.isSupported(format) && 
				(manyToOneDirectedComponent == null || manyToOneDirectedComponent.isSupported(format));
	}

}
