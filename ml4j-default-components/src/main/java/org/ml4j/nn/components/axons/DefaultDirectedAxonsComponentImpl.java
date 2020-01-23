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
package org.ml4j.nn.components.axons;

import java.util.Optional;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of a DefaultChainableDirectedComponent adapter for an
 * Axons instance.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the LHS of this DirectedAxonsComponent.
 * @param <R> The type of Neurons on the RHS of this DirectedAxonsComponent.
 * @param <A> The specific type of Axons wrapped by this DirectedAxonsComponent.
 */
public class DefaultDirectedAxonsComponentImpl<L extends Neurons, R extends Neurons>
		extends DirectedAxonsComponentBase<L, R, Axons<? extends L, ? extends R, ?>>
		implements DirectedAxonsComponent<L, R, Axons<? extends L, ? extends R, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedAxonsComponentImpl.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * @param axons The axons instance wrapped by this DirectedAxonsComponent.
	 */
	public DefaultDirectedAxonsComponentImpl(Axons<? extends L, ? extends R, ?> axons) {
		super(axons);
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			AxonsContext context) {
		LOGGER.debug("Forward propagating through DefaultDirectedAxonsComponentImpl");

		if (neuronsActivation.getFeatureCount() != this.getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}

		AxonsActivation axonsActivation = axons.pushLeftToRight(neuronsActivation, null, context);
		NeuronsActivation output = axonsActivation.getPostDropoutOutput();

		if (output.getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}
		return new DefaultDirectedAxonsComponentActivationImpl<>(this, axonsActivation, context);
	}

	@Override
	public DirectedAxonsComponent<L, R, Axons<? extends L, ? extends R, ?>> dup() {
		return new DefaultDirectedAxonsComponentImpl<>(axons.dup());
	}
	
	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return axons.optimisedFor();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return axons.isSupported(format) && NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}

}
