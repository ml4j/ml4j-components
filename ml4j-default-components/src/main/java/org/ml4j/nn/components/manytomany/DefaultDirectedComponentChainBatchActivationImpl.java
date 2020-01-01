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
package org.ml4j.nn.components.manytomany;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.manytomany.base.DirectedComponentChainBatchActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBatchActivationImpl extends DirectedComponentChainBatchActivationBase implements DefaultDirectedComponentChainBatchActivation{

	
	public DefaultDirectedComponentChainBatchActivationImpl(List<DefaultDirectedComponentChainActivation> activations) {
		super(activations);
	}

	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> outerGradient) {
		
		int index = 0;
		List<Supplier<AxonsGradient>> allAxonsGradients = new ArrayList<>();
		allAxonsGradients.addAll(outerGradient.getTotalTrainableAxonsGradients());
		List<NeuronsActivation> combinedOutput = new ArrayList<>();
		SortedMap<Integer, NeuronsActivation> combinedOutputMap = new TreeMap<>();
		List<ActivationGradientIndex> activationGradients = new ArrayList<>();
		for (DefaultDirectedComponentChainActivation activation : activations) {
			DirectedComponentGradient<NeuronsActivation> grad = new DirectedComponentGradientImpl<>(outerGradient.getOutput().get(index));
			ActivationGradientIndex activationGradient = new ActivationGradientIndex(activation, grad, index);
			activationGradients.add(activationGradient);
			index++;
		}
		
		for (GradientIndex backPropGrad : activationGradients.parallelStream().map(a -> new GradientIndex(a.getActivation().backPropagate(a.getGradient()), a.getIndex())).collect(Collectors.toList())) {
			combinedOutputMap.put(backPropGrad.getIndex(), backPropGrad.getGradient().getOutput());
			//combinedOutput.add(backPropGrad.getGradient().getOutput());
			List<Supplier<AxonsGradient>> backPropAxonsGradients = backPropGrad.getGradient().getTotalTrainableAxonsGradients();
			allAxonsGradients.addAll(backPropAxonsGradients);
		}
		combinedOutput.addAll(combinedOutputMap.values());

		return new DirectedComponentGradientImpl<>(allAxonsGradients, combinedOutput);
	}

	@Override
	public List<? extends ChainableDirectedComponentActivation<List<NeuronsActivation>>> decompose() {
		return Arrays.asList(this);
	}
	
	private class GradientIndex {
		private DirectedComponentGradient<NeuronsActivation>  gradient;
		private int index;
		
		public GradientIndex(DirectedComponentGradient<NeuronsActivation>  gradient, int index) {
			this.gradient = gradient;
			this.index = index;
		}

		public DirectedComponentGradient<NeuronsActivation>  getGradient() {
			return gradient;
		}

		public int getIndex() {
			return index;
		}
	}
	
	private class ActivationGradientIndex {
		
		private DefaultDirectedComponentChainActivation activation;
		private DirectedComponentGradient<NeuronsActivation> gradient;
		private int index;
		
		public ActivationGradientIndex(DefaultDirectedComponentChainActivation activation, DirectedComponentGradient<NeuronsActivation> gradient, int index) {
			this.activation = activation;
			this.gradient = gradient;
			this.index = index;
		}
		
		public int getIndex() {
			return index;
		}

		public DefaultDirectedComponentChainActivation getActivation() {
			return activation;
		}

		public DirectedComponentGradient<NeuronsActivation> getGradient() {
			return gradient;
		}
	}
}
