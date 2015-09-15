/*
 * Disciplina: Algoritmos em grafos
 * Professor: Rodrigo Caetano Rocha
 * Topico: Estruturas de Dados de Digrafos;
 *         Lista de adjacencias.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Trabalho_Pratico_I
{

    /**
     * Classe que implementa um grafo direcionado, ou digrafo.
     */
    public class ListaDigrafo : Digrafo
    {
        private int n;
        private List<int>[] adj;

        /**
         * Construtor de um digrafo de n vertices, sem nenhuma aresta.
         * As arestas devem ser adicionadas posteriormente.
          * Os vertices sao rotulados de 0 ate n-1.
         */
        public ListaDigrafo(int n){
            this.n = n;
            this.adj = new List<int>[n];

            for (int u = 0; u < this.n; u++)
            {
                this.adj[u] = new List<int>();
            }
        }

        /**
         * Adiciona uma nova aresta direcionada do vertice u para v.
         */
        public override void adicionaAresta(int u, int v){
            this.adj[u].Add(v);
        }

        /**
         * Remove uma ocorrencia da aresta direcionada do vertice u para v, caso a mesma exista.
         */
        public override void removeAresta(int u, int v){
            this.adj[u].Remove(v); //remove uma ocorrencia da aresta de u para v.
        }

        /**
         * Verifica se existe uma aresta direcionada do vertice u para v.
         * @return true caso existir a aresta (u,v) ou false caso contrario.
         */
        public override Boolean existeAresta(int u, int v){
            if (this.adj[u].Contains(v))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * Obtem o numero de vertices do digrafo.
        * @return o numero de vertices.
         */
        public override int numVertices(){
            return this.n;
        }

        /**
         * Obtem o numero de arestas direcionadas do digrafo.
        * @return o numero de arestas do digrafo.
         */
        public override int numArestas(){
            int arestas = 0;
            for (int u = 0; u < numVertices(); u++)
            {
                arestas += adj[u].Count();
            }
            return arestas;
        }

        /**
        * Obtem os vizinhos de um dado vertice u.
        * @return conjunto de vizinhos do vertice u.
        */
        public override List<int> vizinhos(int u){
            return adj[u];
        }

        /**
         * Verifica se existe algum loop no Digrafo.
        * @return true caso existir algum loop em qualquer vertice ou false caso contrario.
         */
        public override Boolean existeLoop(){
            //FAZER
            return false; //resposta temporaria
        }

        /**
         * Calcula o grau de saida do vertice u.
        * O grau de saida de um vertice u representa o numero de arestas direcionadas que possuem u como origem.
        * @return grau de saida do vertice u.
         */
        public override int grauSaida(int u){
            //FAZER
            return 0; //resposta temporaria
        }

        /**
         * Calcula o grau de entrada do vertice u.
        * O grau de entrada de um vertice u representa o numero de arestas direcionadas que possuem u como destino.
        * @return grau de entrada do vertice u.
         */
        public override int grauEntrada(int u){
            //FAZER
            return 0; //resposta temporaria
        }

        public void removerArestasParalelas(){
            for (int u = 0; u < numVertices(); u++)
            {
                adj[u] = new List<int>(new HashSet<int>(adj[u]));
            }
        }
    }
}
