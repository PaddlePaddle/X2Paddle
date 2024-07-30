set +x

# use pre-commit 2.17
if ! [[ $(pre-commit --version) == *"2.17.0"* ]]; then
    pip install pre-commit==2.17.0 1>nul
fi

diff_files=$(git diff --name-only --diff-filter=ACMR ${BRANCH})
num_diff_files=$(echo "$diff_files" | wc -l)
echo -e "diff files between pr and ${BRANCH}:\n${diff_files}"

echo "Checking code style by pre-commit ..."
pre-commit run --files ${diff_files};check_error=$?

if test ! -z "$(git diff)"; then
    echo -e '\n************************************************************************************'
    echo -e "These files have been formatted by code format hook. You should use pre-commit to \
format them before git push."
    echo -e '************************************************************************************\n'
    git diff 2>&1
fi

echo -e '\n************************************************************************************'
if [ ${check_error} != 0 ];then
    echo "Your PR code style check failed."
    echo "Please install pre-commit locally and set up git hook scripts:"
    echo ""
    echo "    pip install pre-commit==2.17.0"
    echo "    pre-commit install"
    echo ""
    if [[ $num_diff_files -le 100 ]];then
        echo "Then, run pre-commit to check codestyle issues in your PR:"
        echo ""
        echo "    pre-commit run --files" $(echo ${diff_files} | tr "\n" " ")
        echo ""
    fi
    echo "For more information, please refer to our codestyle check guide:"
    echo "https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/git_guides/codestyle_check_guide_cn.html"
else
    echo "Your PR code style check passed."
fi
echo -e '************************************************************************************\n'

exit ${check_error}
