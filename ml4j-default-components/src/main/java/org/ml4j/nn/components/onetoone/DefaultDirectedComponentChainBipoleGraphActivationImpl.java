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
import java.util.List;

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation for an activation from a
 * DefaultDirectedComponentChainBipoleGraphImpl.
 * 
 * Encapsulates the activations from a forward propagation through a
 * DefaultDirectedComponentChainActivationImpl including the output
 * NeuronsActivation from the RHS of the
 * DefaultDirectedComponentChainActivationImpl.
 * 
 * Also included are the activations from the components within the
 * DefaultDirectedComponentChainActivationImpl - from the one to many, many to
 * one, and batch components.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultDirectedComponentChainBipoleGraphActivationImpl
		extends DefaultDirectedComponentChainBipoleGraphActivationBase
		implements DefaultDirectedComponentChainBipoleGraphActivation {

	private OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation;
	private DefaultDirectedComponentChainBatchActivation parallelChainsActivation;
	private ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation;
	private boolean originalInputIsImmutable;

	/**
	 * Constructor for a DefaultDirectedComponentBipoleGraphActivationImpl to be
	 * used when the originating DefaultDirectedComponentBipoleGraph contains a
	 * single edge only, for optimal efficiency.
	 * 
	 * @param defaultDirectedComponentChainBipoleGraph The originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param parallelChainsActivation                 The activation from the
	 *                                                 nested
	 *                                                 DefaultDirectedComponentChainBatch
	 *                                                 within the originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param output                                   The output from the
	 *                                                 originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 */
	public DefaultDirectedComponentChainBipoleGraphActivationImpl(
			DefaultDirectedComponentChainBipoleGraph defaultDirectedComponentChainBipoleGraph,
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation, NeuronsActivation output,
			boolean originalInputIsImmutable) {
		super(defaultDirectedComponentChainBipoleGraph, output);
		if (defaultDirectedComponentChainBipoleGraph.getEdges().getComponents().size() != 1) {
			throw new IllegalArgumentException(
					"This constructor should only be used when the DefaultDirectedComponentChainBatchcontains a single edge");
		}
		if (parallelChainsActivation.getActivations().size() != 1) {
			throw new IllegalArgumentException(
					"This constructor should only be used when the DefaultDirectedComponentChainBatchActivation contains a single nested activation");
		}
		this.parallelChainsActivation = parallelChainsActivation;
		this.originalInputIsImmutable = originalInputIsImmutable;
	}

	/**
	 * Constructor for a DefaultDirectedComponentBipoleGraphActivationImpl
	 * 
	 * @param defaultDirectedComponentChainBipoleGraph The originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param oneToManyDirectedComponentActivation     The activation from the
	 *                                                 nested
	 *                                                 OneToManyDirectedComponent
	 *                                                 within the originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param parallelChainsActivation                 The activation from the
	 *                                                 nested
	 *                                                 DefaultDirectedComponentChainBatch
	 *                                                 within the originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param manyToOneDirectedComponentActivation     The activation from the
	 *                                                 nested
	 *                                                 ManyToOneDirectedComponent
	 *                                                 within the originating
	 *                                                 DefaultDirectedComponentBipoleGraph.
	 * @param originalInputIsImmutable
	 */
	public DefaultDirectedComponentChainBipoleGraphActivationImpl(
			DefaultDirectedComponentChainBipoleGraph defaultDirectedComponentChainBipoleGraph,
			OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation,
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation,
			ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation,
			boolean originalInputIsImmutable) {
		super(defaultDirectedComponentChainBipoleGraph, manyToOneDirectedComponentActivation.getOutput());
		this.oneToManyDirectedComponentActivation = oneToManyDirectedComponentActivation;
		this.manyToOneDirectedComponentActivation = manyToOneDirectedComponentActivation;
		this.parallelChainsActivation = parallelChainsActivation;
		this.originalInputIsImmutable = originalInputIsImmutable;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {

		if (parallelChainsActivation.getActivations().size() == 1) {
			DirectedComponentGradient<List<NeuronsActivation>> gradient = new DirectedComponentGradientImpl<>(
					outerGradient.getTotalTrainableAxonsGradients(), Arrays.asList(outerGradient.getOutput()));
			DirectedComponentGradient<List<NeuronsActivation>> edgesGradients = parallelChainsActivation
					.backPropagate(gradient);
			return new DirectedComponentGradientImpl<>(edgesGradients.getTotalTrainableAxonsGradients(),
					edgesGradients.getOutput().get(0));

		} else {
			DirectedComponentGradient<List<NeuronsActivation>> manyToOneActivation = manyToOneDirectedComponentActivation
					.backPropagate(outerGradient);
			DirectedComponentGradient<List<NeuronsActivation>> edgesGradients = parallelChainsActivation
					.backPropagate(manyToOneActivation);
			DirectedComponentGradient<NeuronsActivation> result = oneToManyDirectedComponentActivation
					.backPropagate(edgesGradients);
			edgesGradients.getOutput().forEach(a -> a.close());
			manyToOneActivation.getOutput().forEach(a -> a.close());
			return result;
		}
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		if (completedLifeCycleStage == DirectedComponentActivationLifecycle.FORWARD_PROPAGATION) {
			if (!originalInputIsImmutable) {
				if (oneToManyDirectedComponentActivation != null) {
					// oneToManyDirectedComponentActivation.getOutput().forEach(a -> a.close());
				}
			}
			if (manyToOneDirectedComponentActivation != null) {
				if (manyToOneDirectedComponentActivation.getOutput().isImmutable()) {
					manyToOneDirectedComponentActivation.getOutput().close();
				}
			}
		}
		parallelChainsActivation.close(completedLifeCycleStage);
	}

}
