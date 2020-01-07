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

import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation for an activation from a DefaultBatchNormDirectedAxonsComponent.
 *   
 * @author Michael Lavelle
 * 
 * @param <A> The type of Axons within the batch norm DirectedAxonsComponent from which this activation originated
 */
public class DefaultBatchNormDirectedAxonsComponentActivationImpl<A extends Axons<?, ?, ?>> extends DirectedAxonsComponentActivationBase<A> implements DirectedAxonsComponentActivation {
	
	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultBatchNormDirectedAxonsComponentActivationImpl.class);

	public DefaultBatchNormDirectedAxonsComponentActivationImpl(DirectedAxonsComponent<?, ?, A> axonsComponent, AxonsActivation axonsActivation, AxonsContext axonsContext) {
		super(axonsComponent, axonsActivation, axonsContext);
	}

	@Override
	public float getTotalRegularisationCost() {
		// TODO
		return 0;
	}

	@Override
	protected DirectedComponentGradientImpl<NeuronsActivation> createBackPropagatedGradient(AxonsActivation axonsActivation,
			List<Supplier<AxonsGradient>> previousAxonsGradients, Supplier<AxonsGradient> thisAxonsGradient) {
		return new DirectedComponentGradientImpl<>(previousAxonsGradients, thisAxonsGradient, axonsActivation.getPostDropoutOutput());
	}

	@Override
	protected Optional<AxonsGradient> getCalculatedAxonsGradient(AxonsActivation rightToLeftAxonsGradientActivatoin) {
		
		if (directedAxonsComponent.getAxons().isTrainable(axonsContext)) {
			// TODO
			throw new UnsupportedOperationException("Not implemented yet");
			
		} else {
			return Optional.empty();
		}
	}
}
