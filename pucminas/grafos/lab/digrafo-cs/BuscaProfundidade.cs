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

    public abstract class BuscaProfundidade : Busca
    {
        private Digrafo g;
        private int vertice_Origem;

        public BuscaProfundidade(Digrafo g, int verticeOrigem) : base(g, verticeOrigem)
        {
        }

        protected override void realizaBusca()
        {
            //FAZER
        }

        public override Boolean existeCaminhoPara(int verticeDestino)
        {
            //FAZER
            return false;
        }

        public override List<int> caminhoPara(int verticeDestino)
        {
            //FAZER
            return null;
        }

        public override double pesoCaminhoPara(int verticeDestino)
        {
            //FAZER
            return double.NaN;
        }
    }
}
