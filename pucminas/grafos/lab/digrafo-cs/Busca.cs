/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Trabalho_Pratico_I
{

    public abstract class Busca
    {
        private Digrafo g;
        private int vertice_Origem;

        public Busca(Digrafo g, int verticeOrigem)
        {
            this.g = g;
            this.vertice_Origem = verticeOrigem;
            realizaBusca();
        }

        public Digrafo digrafo()
        {
            return this.g;
        }

        public int verticeOrigem()
        {
            return this.vertice_Origem;
        }

        protected abstract void realizaBusca();

        public abstract Boolean existeCaminhoPara(int verticeDestino);

        public abstract List<int> caminhoPara(int verticeDestino);

        public abstract double pesoCaminhoPara(int verticeDestino);
    }
}
