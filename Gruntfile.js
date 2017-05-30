module.exports = function(grunt) {

  // Project configuration.
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    
    watch: {

    },

    uglify: {
      my_target: {
        files: {
          'public/js/typed.min.js': ['public/js/typed.js'],
          'public/js/scripts.min.js': ['public/js/scripts.js']
        }
      }
    },
    
    cssmin: {
      target: {
        files: {
        'public/css/release.css': ['public/css/lanyon.css', 'public/css/poole.css','public/css/syntax.css']
        }
      }
    },
    
    htmlmin: {

    }
    
  });

  // Load npms
  grunt.loadNpmTasks('grunt-contrib-uglify');
  grunt.loadNpmTasks('grunt-contrib-watch');
  grunt.loadNpmTasks('grunt-contrib-cssmin');
  grunt.loadNpmTasks('grunt-contrib-htmlmin');
  grunt.loadNpmTasks('grunt-jekyll');

  // Default task(s).
  grunt.registerTask('default', ['uglify','cssmin']);

};